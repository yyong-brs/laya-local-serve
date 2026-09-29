# Laya 内网部署指南（本地权重 + Jev 兼容 HTTP 服务）

本指南说明如何在不依赖外网（Hugging Face Hub）的前提下，使用 NVIDIA GPU 通过 Docker 部署
`laya-multilingual` 模型，并以官方同款的 Jev 兼容 HTTP 接口（`POST /v1/systemone`、`GET /health`）对外提供服务。

---

## 1. 背景与关键结论

需求：在内网（无外网）用 Docker 部署 `laya-multilingual`，基于 NVIDIA 显卡，加载手动挂载的本地权重，提供 Jev 兼容的 HTTP 接口。

经源码核实，有几个必须注意的点：

- **官方 `laya-serve` 不会读取本地权重路径。** 环境变量 `LAYA_MODEL_PATH` 只被一次性脚本
  `examples/docker/quickstart.py` 消费；HTTP 服务 `laya-serve` 的 `build_router()` 始终把
  checkpoint 解析到 Hugging Face Hub。因此直接 `laya-serve + 挂载本地权重` 会在内网下回退去
  HF Hub 下载而失败。
- **HTTP 接口天然 Jev 兼容。** `laya/serve.py` 暴露 `POST /v1/systemone` 与 `GET /health`，
  输出符合 TypeSafe Jev 线协议。
- **`Agent` 直接支持本地目录权重。** `agent.py` 中存在 `os.path.exists(model_dir)` 分支，
  走本地加载路径，不触发 `snapshot_download`，因此本地权重完全可离线使用。

基于以上结论，本方案**不修改上游 `serve.py`**，而是新增一个挂载式启动器，把本地目录权重构建成
一个 `Agent` 并同时挂到多个路由别名，再复用官方 `create_app()` 暴露同样的接口。

---

## 2. 新增文件

| 文件 | 作用 |
|---|---|
| `docker/serve_local.py` | 本地权重启动器：从 `LAYA_LOCAL_MODEL_PATH` 目录构建 **1 个** `Agent`，通过 `Router.attach` 同时挂到 `english` / `multilingual` / `typed-decisions` 三个路由别名（共享一份权重，避免按别名重复加载占用双倍显存），再复用官方 `laya.serve.create_app()` 暴露同样的 `POST /v1/systemone` + `GET /health`。 |
| `compose.local-serve.yaml` | （可选）Compose 覆盖文件，把 `laya-serve` 的启动命令替换为 `serve_local.py`，并挂载权重目录与启动器脚本。 |
| `Dockerfile_new` | 把 Laya 代码与 `serve_local.py` **烤进镜像**，运行时只需挂载权重目录；镜像默认 `LAYA_DEVICE=cuda`、`HF_HUB_OFFLINE=1`，`CMD` 已内置启动命令。 |

> 推荐直接使用 `Dockerfile_new` 路线（运行时最简洁：只挂权重）。`compose.local-serve.yaml`
> 适合已经使用官方 compose 栈的场景，见文末附录。

---

## 3. 构建镜像（Dockerfile_new）

代码已烤进镜像，构建时需要联网拉取 `torch` 与依赖（构建机或私有镜像仓库需可访问 PyPI / PyTorch 源）。

> **Dockerfile_new 基于官方多 stage 写法**，仅做了三处增量：把 `docker/serve_local.py` 烤进镜像、
> 镜像默认 `LAYA_DEVICE=cuda` / `HF_HUB_OFFLINE=1` / `LAYA_CUDA_AMP=fp16` / `LAYA_HOST=0.0.0.0`、
> 并把默认 `CMD` 设为启动本地权重服务。它仍是一道 `COPY --from=build /opt/venv` 的多 stage 构建——
> **在 WSL2 默认 2 GB 内存下，这道跨 stage 拷贝 + `pip install torch` 解压会 OOM 报 `EOF`**。
> 因此推荐用 GitHub Actions（见 3.4）在 CI 上构建，CI runner 内存充足，可稳定通过。

### 3.1 前置条件（仅当本机 WSL2 直接构建时需要）：WSL2 必须提高内存

若你**在本机 WSL2 上直接 `docker build`**（不走 CI），WSL2 默认只有 2 GB 内存，`pip install torch`
解压 CUDA 全家桶本身仍需约 **4 GB 峰值内存**，所以**构建前必须先提内存**，否则照样 OOM 报 `EOF`：

- **有 Docker Desktop**（任务栏鲸鱼图标）：编辑 `%APPDATA%\Docker\settings.json`，把 `memoryMiB`
  改成 `12288`（=12 GB）或更大；保存后**右键托盘图标 Quit 彻底退出**再重开。
- **纯 WSL2 Docker Engine / 想调全局**：编辑 `C:\Users\<你>\.wslconfig`，写
  `[wsl2] memory=12GB swap=4GB`，然后在 PowerShell（管理员）执行 `wsl --shutdown` 生效。

改完用 `docker info` 确认 **Memory** 行约 12 GiB，再开始构建。

### 3.2 构建命令

```bash
# 主流 NVIDIA 显卡（默认 cu128 + torch 2.11.0）
docker build -f Dockerfile_new -t laya-local-serve:latest .

# DGX Spark (GB10 / arm64) 改用 cu130 + torch 2.14.0（2.14.0 在 cu130 源上存在）
docker build --build-arg TORCH_INDEX=cu130 --build-arg TORCH_VERSION=2.14.0 -f Dockerfile_new -t laya-local-serve:latest .
```

**完全离线（内网）场景**：在能联网的机器上构建后导出镜像，再拷到内网机加载：

```bash
docker save laya-local-serve:latest -o laya-local-serve.tar
# 将 tar 拷到内网机后执行：
docker load -i laya-local-serve.tar
```

> 注意：`Dockerfile_new` 默认 `TORCH_INDEX=cu128`、`TORCH_VERSION=2.11.0`。cu128 源上 torch
> 最高只到 2.11.0（2.14.0 未在 cu128 发布），因此官方 `Dockerfile` 的默认 `2.14.0` 在 cu128 下会失败；
> 本文件已将其降到 2.11.0 以匹配 cu128 源。DGX Spark 的 cu130 源才有 2.14.0。

### 3.3 关于缓存挂载（重要：它不是"断点续传"）

`Dockerfile_new` 里两道 `pip install` 都挂了 `--mount=type=cache,target=/root/.cache/pip`。
**它的作用仅仅是：当某一步真正跑完成时，把已下载的 wheel 留在缓存里，下次同一步重跑时不必重新下载。**
它**不是失败构建的断点续传**：

- 构建在某一层失败 → 该层**不会提交**，缓存挂载内容通常被 BuildKit GC 掉 → 下次重跑会从下载重新开始
  （你第二次构建又从头下 `nvidia-cudnn` 就是因为这个）。
- 因此"失败应该接着上次继续"这个预期不成立。**真正能避免重来的办法是：让每一步都跑成功**——
  也就是先把上面的内存提到 8 GB+，让 `pip install torch` 这次能完整跑完并提交。一旦成功过一次，
  之后改 `serve_local.py` 再构建，前面所有层都是 `CACHED`，只重跑最后改动的那层，非常快。

建议首次构建加上 `--progress=plain` 看清楚死在哪一步：

```bash
docker build --progress=plain -f Dockerfile_new -t laya-local-serve:latest .
```

### 3.4 用 GitHub Actions 在 CI 上构建（推荐，规避本地 OOM）

仓库已包含 `.github/workflows/build-laya-local-serve.yml`：推送代码到 `main` 分支（或改动
`Dockerfile_new` / `docker/` / `laya/` / `pyproject.toml` / `setup.py`）即自动触发，用
`docker/build-push-action` 在 `ubuntu-latest` runner 上构建，并推送到 **GHCR**
（`ghcr.io/<你的账户>/laya-local-serve:latest` 与 `:${{ github.sha }}`）。也可在 Actions 页面点
`workflow_dispatch` 手动触发，并覆盖 `TORCH_INDEX` / `TORCH_VERSION`（如 DGX Spark 用
`cu130` + `2.14.0`）。

CI runner 内存充足，多 stage 的 `COPY --from=build` 与 `pip install torch` 都能稳定跑过，无需再调
WSL2 内存。

构建完成后，在能联网的机器上拉取并导出，再拷到内网机加载：

```bash
docker pull ghcr.io/<你的账户>/laya-local-serve:latest
docker save ghcr.io/<你的账户>/laya-local-serve:latest -o laya-local-serve.tar
# 将 tar 拷到内网机后执行：
docker load -i laya-local-serve.tar
```

> 注意：镜像含 torch cu128 + CUDA 运行时，体积约 8 GB+，GHCR 免费额度有限，注意清理旧 tag。

> **GHCR 私有包跨仓库拉取报错 `unauthorized`**：`yyong-brs/laya-local-serve` 是私有仓库，
> 其 GHCR 包默认也私有。若在**另一个** GitHub 项目的 flow 里用该项目自己的 `GITHUB_TOKEN`
> 登录 `ghcr.io` 后 `docker pull`，会报 `unauthorized`——因为该 token 只限当前仓库、读不到
> 另一仓库的私有包（登录能过、拉取被拒）。两种解法：① 把 GHCR 包可见性改为 **Public**
> （GitHub → Packages → 该包 → Settings），改完免登录直接 `docker pull`；② 保持私有则在另一个
> 项目 Secrets 中放一个勾了 `read:packages` 的 PAT，用
> `echo "$PAT" | docker login ghcr.io -u yyong-brs --password-stdin`。
> 国内直接拉 GHCR 较慢，推荐改用下面的阿里云 ACR。

### 3.5 国内从阿里云 ACR 拉取（推荐）

工作流在推送到 GHCR 之后，会自动把镜像**同构件复制**到阿里云容器镜像服务 ACR
（`docker buildx imagetools create` 仅复制 manifest+blobs，不二次编译）。该步骤通过 Secrets
控制，**未设置阿里云密钥时自动跳过**，不影响现有 GHCR 推送。

启用步骤：

1. 在 GitHub 仓库 **Settings → Secrets and variables → Actions → New repository secret**
   添加以下密钥（仅当 `ALIYUN_USERNAME` 存在时才触发阿里云推送）：

   | Secret 名 | 含义 | 默认值 |
   |---|---|---|
   | `ALIYUN_USERNAME` | 阿里云账号名 / RAM 子账号 | （无 → 跳过阿里云步骤） |
   | `ALIYUN_PASSWORD` | 阿里云密码 / ACR 专用登录密码 | —— |
   | `ALIYUN_REGISTRY` | ACR 地域域名（可选） | `registry.cn-hangzhou.aliyuncs.com` |
   | `ALIYUN_NAMESPACE` | ACR 命名空间（可选） | 仓库 owner（`yyong-brs`） |

   > 地域域名默认杭州；若用其他地域请设 `ALIYUN_REGISTRY`，例如上海
   > `registry.cn-shanghai.aliyuncs.com`、深圳 `registry.cn-shenzhen.aliyuncs.com`。

2. 推送 `main`（或改动 `Dockerfile_new` 等受监控文件）触发构建，Actions 日志出现
   `Copy image to Aliyun ACR` 即复制成功。

3. 在国内部署机上拉取（需先在阿里云 ACR 把该镜像设为**公开**，或给部署机 RAM 读权限）：

   ```bash
   docker login <ALIYUN_REGISTRY> -u <ALIYUN_USERNAME> -p <ALIYUN_PASSWORD>
   docker pull <ALIYUN_REGISTRY>/<ALIYUN_NAMESPACE>/laya-local-serve:latest
   docker save <ALIYUN_REGISTRY>/<ALIYUN_NAMESPACE>/laya-local-serve:latest -o laya-local-serve.tar
   # 拷到内网机后：docker load -i laya-local-serve.tar
   ```

---

## 4. 运行容器

镜像已将 Laya 代码与本地权重启动器 `serve_local.py` 烤进内部，并把启动命令
`python /opt/laya/serve_local.py` 设为容器的默认 `CMD`（由镜像 `ENTRYPOINT` 的
`entrypoint.py` 负责加载 `_FILE` 形式密钥后转交执行）。因此 **`docker run` 末尾
不需要再写启动命令**——下面的命令就是完整且自包含的启动方式。

运行时只需做两件事：**挂载权重目录** + **通过 `-e` 直接传入环境变量**。

### 4.1 启动命令（环境变量直接写在命令中）

```bash
docker run -d --name laya-serve --gpus all \
  -p 8000:8000 \
  -v /abs/path/to/laya-multilingual:/models/laya-multilingual:rw \
  -e LAYA_LOCAL_MODEL_PATH=/models/laya-multilingual \
  -e LAYA_DEVICE=cuda \
  -e HF_HUB_OFFLINE=1 \
  -e LAYA_CUDA_AMP=fp16 \
  -e LAYA_HOST=0.0.0.0 \
  -e LAYA_PORT=8000 \
  -e LAYA_API_KEY='你的强密钥' \
  laya-local-serve:latest
```

> 说明：上面把 `LAYA_DEVICE` / `HF_HUB_OFFLINE` / `LAYA_CUDA_AMP` / `LAYA_HOST` /
> `LAYA_PORT` 都显式写成 `-e`，是为了**在命令中直接看清并可控**（这些在镜像里已是默认值）。
> 若接受默认行为，可只保留 `LAYA_LOCAL_MODEL_PATH`（必填）与 `LAYA_API_KEY`（建议设置）。
> 镜像的容器启动命令 `python /opt/laya/serve_local.py` 已内置，**不要**在命令末尾重复追加它。

### 4.2 参数说明

- `--gpus all`：启用 NVIDIA GPU（需宿主机安装 NVIDIA Container Toolkit）。老版本 Docker 不支持
  `--gpus` 时改用 `--runtime=nvidia`（需 `nvidia-docker2`）。
- `-v ...:/models/laya-multilingual:rw`：将已下载好的 `laya-multilingual` **完整目录**挂载进容器。
  目录内需包含 `rl_agent_config.json`、`model.safetensors`、`tokenizer/`、`encoder/` 等权重文件；
  `:rw` 是因为首次加载会重写 `tokenizer_config.json`。
- `-e LAYA_LOCAL_MODEL_PATH`：**必填**，告知启动器权重目录；缺失会直接退出。
- `-e LAYA_API_KEY`：设置后 `/v1/systemone` 要求 Bearer 鉴权；不需要鉴权则去掉该行
  （`/health` 始终免鉴权）。也可改用 `LAYA_API_KEY_FILE=/run/secrets/xxx` 挂载密钥文件，
  由 `entrypoint.py` 自动读取并注入环境变量。
- `-e LAYA_GPU_ID`：多卡时指定 GPU 编号（默认 `0`）。

---

## 5. 验证

```bash
# 1) 查看启动日志，确认 device=cuda、loaded 含 multilingual/english
docker logs -f laya-serve

# 2) 健康检查（始终免鉴权）
curl -s localhost:8000/health

# 3) 发送一次 Jev 兼容推理请求（设了 LAYA_API_KEY 时带 Bearer）
curl -s localhost:8000/v1/systemone \
  -H 'authorization: Bearer 你的密钥' \
  -H 'content-type: application/json' \
  --data @/abs/path/to/laya/examples/docker/request.json
```

成功时 `/health` 返回中 `device` 为 `cuda`、`loaded` 列表包含 `multilingual`；`/v1/systemone`
返回 `choice` / `score` / `noul` 等字段的 Jev 兼容答案。

---

## 6. 注意事项

- **权重目录权限**：容器以 UID/GID 10001（`laya` 用户）运行，首次加载会重写
  `tokenizer_config.json`，因此挂载已用 `:rw`。若宿主机 ACL 拦截写入，先执行
  `chmod -R a+rwX /abs/path/to/laya-multilingual`。
- **离线保障**：`HF_HUB_OFFLINE=1` 为兜底；本地目录加载路径本身就不触发 `snapshot_download`，
  双重保险。
- **显存占用**：单个 `multilingual`（约 322M）常驻，三个路由别名共享同一 `Agent` 实例，不翻倍。
- **暴露面**：镜像默认 `LAYA_HOST=0.0.0.0`，仅限可信内网；对外务必在前面加 TLS 反向代理，
  并配合 `LAYA_API_KEY` 鉴权。
- **多卡选择**：如需指定 GPU，运行时加 `-e LAYA_GPU_ID=1`（镜像默认 `0`）。

---

## 7. 构建失败排障（EOF / OOM）

构建过程中若报 `failed to receive status: rpc error: code = Unavailable desc = error reading from server: EOF`，
这是 **BuildKit 进程被 OOM 杀掉 / 与守护进程连接断开**，不是代码错误。按以下顺序排查：

1. **先确认失败发生在哪一步**（用 `--progress=plain` 重跑）：
   - 死在 `Installing collected packages: ... torch` / `COPY --from=build /opt/venv` → 内存不足。
     本镜像为官方多 stage 写法，跨 stage 的 `COPY --from=build /opt/venv` 与 `pip install torch`
     解压都会吃大量内存。若你在**本机 WSL2 直接构建**仍死在这一步，说明 WSL2 内存仍不够，
     **回到 3.1 把内存提到 8 GB+**；或干脆走 3.4 的 CI 构建（runner 内存充足，可稳定通过）。
   - 死在正在 `Downloading ... .whl` 且速度很慢/中断 → 网络抖动，调完内存后重跑即可
     （成功过的层会 `CACHED`，不会重下）。
2. **磁盘是否写满**：`df -h` 看 Docker 所在盘；或用 `docker system df` 看 BuildKit 缓存占用。
   若接近满，先 `docker system prune -a` 释放（只清无引用的镜像/缓存，不影响已构建成果）。
3. **WSL2 GUI 无法设置内存时**：按 3.1 改 `%APPDATA%\Docker\settings.json` 的 `memoryMiB`
   或 `C:\Users\<你>\.wslconfig` 的 `memory=12GB`，改完务必**彻底退出 Docker Desktop 再重开**
   （或 `wsl --shutdown`）。

> 经验值：torch cu128 + `nvidia-*` CUDA 运行时解压后约 6–8 GB，构建期峰值内存 4 GB 左右。
> WSL2 默认 2 GB 必崩；给到 8–12 GB 后构建基本稳定通过。

---

## 附录 A：Compose 覆盖方案（compose.local-serve.yaml）

若你使用官方 compose 栈，可叠加 `compose.local-serve.yaml`（CUDA 由 `compose.cuda.yaml` 提供，
**必须放在最后**）：

```bash
export LAYA_CHECKPOINT_PATH=/abs/path/to/laya-multilingual
docker compose -f compose.yaml -f compose.http.yaml -f compose.cuda.yaml -f compose.local-serve.yaml up --build laya-serve
```

该方案需要在运行时额外挂载 `serve_local.py`（`:ro`），并显式传入 `LAYA_DEVICE=cuda`、
`HF_HUB_OFFLINE=1` 等环境变量；其余行为与 `Dockerfile_new` 路线一致。

---

## 附录 B：备选路线（修改上游）

若愿意修改上游 `serve.py`，可在 `build_router()` 中读取 `LAYA_LOCAL_MODEL_PATH` 并注入
`Router(models=...)`，从而省去挂载启动器脚本。当前默认方案选择"挂载脚本 + 不碰上游"，更易随
仓库同步升级。
