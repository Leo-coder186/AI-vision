# LLM 神经网络可视化

基于 Three.js 的大语言模型计算过程交互式 3D 可视化演示。

![预览](public/favicon.svg)

## 功能特性

- **Token 可视化** - 输入文本渲染为粒子点云
- **Embedding 空间** - 粒子展开到高维向量空间
- **Attention 动画** - QKV 矩阵运算的粒子连线效果
- **FFN 激活** - 多层感知机的非线性激活展示
- **实时 API 响应** - 接入 MiniMax 大模型获取回答

## 快速开始

### 环境要求

- Node.js >= 20.0

### 安装

```bash
npm install
```

### 配置

复制环境变量文件并填入你的 MiniMax API Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```
MINIMAX_API_KEY=你的API密钥
```

### 启动

```bash
node --env-file=.env server.mjs
```

访问 http://localhost:3000

## 开发模式

```bash
npm run dev
```

## 项目结构

```
├── src/
│   ├── main.ts      # 主程序 - 3D 场景、动画、API 调用
│   └── style.css    # 样式文件
├── server.mjs       # Node.js HTTP 服务器（代理 API 请求）
├── test-api.js      # API 测试脚本
├── index.html       # 入口 HTML
└── public/          # 静态资源
```

## API 代理

服务器代理 MiniMax API：

```
POST /api/chat
Content-Type: application/json

{"question": "你的问题"}
```

## 技术栈

- TypeScript + Vite
- Three.js + GSAP
- Node.js HTTP Server

## License

MIT
