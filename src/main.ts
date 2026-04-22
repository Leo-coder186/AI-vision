import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import gsap from 'gsap';

type TextCoordinateResult = {
  coords: THREE.Vector3[];
  tokens: number[];
  numTokens: number;
};

type ApiTextBlock = {
  type: string;
  text?: string;
};

type ApiResponse = {
  content?: ApiTextBlock[];
};

function getRequiredElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing required element: #${id}`);
  }

  return element as T;
}

const appDiv = getRequiredElement<HTMLDivElement>('app');
const hudStage = getRequiredElement<HTMLSpanElement>('stage-text');
const hudDim = getRequiredElement<HTMLSpanElement>('dim-text');
const hudParams = getRequiredElement<HTMLSpanElement>('params-text');
const logContainer = getRequiredElement<HTMLDivElement>('log-container');
const inputElement = getRequiredElement<HTMLInputElement>('user-input');
const submitBtn = getRequiredElement<HTMLButtonElement>('submit-btn');
const stageOverlay = getRequiredElement<HTMLDivElement>('stage-overlay');
const stageBigText = getRequiredElement<HTMLElement>('stage-big-text');
const stageSubText = getRequiredElement<HTMLParagraphElement>('stage-sub-text');
const aiResponseContainer = getRequiredElement<HTMLDivElement>('ai-response-container');
const aiResponseText = getRequiredElement<HTMLDivElement>('ai-response-text');
const attentionCalcPanel = getRequiredElement<HTMLDivElement>('attention-calc-panel');
const qkvLog = getRequiredElement<HTMLDivElement>('qkv-log');
const ffnCalcPanel = getRequiredElement<HTMLDivElement>('ffn-calc-panel');
const ffnLog = getRequiredElement<HTMLDivElement>('ffn-log');

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x050510, 0.0004);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 5000);
camera.position.set(0, 0, 800);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x050510, 1);
appDiv.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.maxDistance = 4000;
controls.target.set(0, 0, 0);

const renderScene = new RenderPass(scene, camera);
const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.5, 0.4, 0.85);
bloomPass.threshold = 0;
bloomPass.strength = 1.3;
bloomPass.radius = 0.6;

const composer = new EffectComposer(renderer);
composer.addPass(renderScene);
composer.addPass(bloomPass);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  composer.setSize(window.innerWidth, window.innerHeight);
});

const colorBlue = new THREE.Color(0x00f3ff);
const colorPurple = new THREE.Color(0x9d00ff);
const colorGreen = new THREE.Color(0x00ff66);
const colorQuery = new THREE.Color(0xff3366);
const colorKey = new THREE.Color(0x33ff66);
const colorValue = new THREE.Color(0x33ccff);
const colorBurst = new THREE.Color(0xffff00);
const cameraLookTarget = new THREE.Vector3();
const mouse = new THREE.Vector2();
const mouseWorld = new THREE.Vector3();
const raycaster = new THREE.Raycaster();
let scatterFactor = 0;

let pCount = 0;
let sourceTokens: number[] = [];
let maxTokens = 1;

let posTokens: THREE.Vector3[] = [];
let posEmbeds: THREE.Vector3[] = [];
let posAttn: THREE.Vector3[] = [];
let posFFN: THREE.Vector3[] = [];
let posOut: THREE.Vector3[] = [];
let currentPositions = new Float32Array();

let geometry: THREE.BufferGeometry | null = null;
let particles: THREE.Points | null = null;
let attnLinesGeom: THREE.BufferGeometry | null = null;
let attnLinesMesh: THREE.LineSegments | null = null;

const stateInfo = { progress: 0 };

let activeTimeline: gsap.core.Timeline | null = null;
let activeCameraTimeline: gsap.core.Timeline | null = null;
let typingInterval: number | null = null;
let stageOverlayTimeout: number | null = null;
let globalAiResponse = '';
let hasTriggeredTyping = false;
let isResponseReady = false;

function getTextCoordinates(text: string, scale = 1): TextCoordinateResult {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  if (!ctx) {
    return { coords: [], tokens: [], numTokens: 0 };
  }

  canvas.width = 1024;
  canvas.height = 256;

  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.font = 'bold 120px "Noto Sans SC", monospace';
  ctx.fillStyle = '#ffffff';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);

  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const coords: THREE.Vector3[] = [];
  const tokens: number[] = [];
  const activeXs: number[] = [];
  const gap = 4;

  for (let y = 0; y < canvas.height; y += gap) {
    for (let x = 0; x < canvas.width; x += gap) {
      const index = (y * canvas.width + x) * 4;
      if (imgData[index] > 128) {
        const pX = (x - canvas.width / 2) * scale;
        const pY = -(y - canvas.height / 2) * scale;
        coords.push(
          new THREE.Vector3(
            pX + (Math.random() - 0.5) * 2,
            pY + (Math.random() - 0.5) * 2,
            (Math.random() - 0.5) * 5,
          ),
        );
        activeXs.push(pX);
      }
    }
  }

  if (activeXs.length === 0) {
    return { coords, tokens, numTokens: 0 };
  }

  const minX = Math.min(...activeXs);
  const maxX = Math.max(...activeXs);
  const width = maxX - minX;
  const numTokens = Math.max(1, Math.ceil(Math.max(width, 1) / 80));

  for (let i = 0; i < coords.length; i += 1) {
    const x = coords[i].x;
    const normalized = width === 0 ? 0 : (x - minX) / width;
    const tokenIndex = Math.min(numTokens - 1, Math.floor(normalized * numTokens));
    tokens.push(tokenIndex);
  }

  return { coords, tokens, numTokens };
}

function updateTargetOutput(text: string) {
  if (pCount === 0) {
    return;
  }

  const outData = getTextCoordinates(text, 1.2);
  const fallbackOutData = getTextCoordinates('数据映射完成', 1.2);
  const targetData = outData.coords.length > 0 ? outData : fallbackOutData;

  posOut = [];
  for (let i = 0; i < pCount; i += 1) {
    const len = targetData.coords.length;
    const sourcePoint = targetData.coords[i % len] ?? new THREE.Vector3();
    posOut.push(new THREE.Vector3(sourcePoint.x, sourcePoint.y, -3000 + sourcePoint.z));
  }
}

function disposeParticleScene() {
  if (particles && geometry) {
    scene.remove(particles);
    geometry.dispose();
    (particles.material as THREE.Material).dispose();
  }

  if (attnLinesMesh && attnLinesGeom) {
    scene.remove(attnLinesMesh);
    attnLinesGeom.dispose();
    (attnLinesMesh.material as THREE.Material).dispose();
  }

  geometry = null;
  particles = null;
  attnLinesGeom = null;
  attnLinesMesh = null;
}

function initOrUpdateParticles(inputText: string): boolean {
  disposeParticleScene();

  const inputData = getTextCoordinates(inputText, 1);
  pCount = inputData.coords.length;

  if (pCount === 0) {
    return false;
  }

  sourceTokens = inputData.tokens;
  maxTokens = inputData.numTokens;
  posTokens = inputData.coords;

  posEmbeds = [];
  for (let i = 0; i < pCount; i += 1) {
    const point = posTokens[i];
    posEmbeds.push(
      new THREE.Vector3(
        point.x + (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 400,
        (Math.random() - 0.5) * 20,
      ),
    );
  }

  posAttn = [];
  const radius = 300;
  for (let i = 0; i < pCount; i += 1) {
    const tokenIdx = sourceTokens[i];
    const angle = (tokenIdx / Math.max(maxTokens, 1)) * Math.PI * 2;
    const qkvKind = i % 3;
    const ringOffset = radius + (qkvKind - 1) * 40;

    posAttn.push(
      new THREE.Vector3(
        Math.cos(angle) * ringOffset,
        (Math.random() - 0.5) * 150,
        -800 + Math.sin(angle) * ringOffset,
      ),
    );
  }

  posFFN = [];
  for (let i = 0; i < pCount; i += 1) {
    const rad = 500 * Math.cbrt(Math.random());
    const theta = Math.random() * 2 * Math.PI;
    const phi = Math.acos(2 * Math.random() - 1);

    posFFN.push(
      new THREE.Vector3(
        rad * Math.sin(phi) * Math.cos(theta),
        rad * Math.sin(phi) * Math.sin(theta),
        -1800 + rad * Math.cos(phi),
      ),
    );
  }

  updateTargetOutput('等待响应...');

  geometry = new THREE.BufferGeometry();
  const posArray = new Float32Array(pCount * 3);
  const colorsArray = new Float32Array(pCount * 3);
  const sizeArray = new Float32Array(pCount);
  currentPositions = new Float32Array(pCount * 3);

  for (let i = 0; i < pCount; i += 1) {
    const point = posTokens[i];
    const i3 = i * 3;

    posArray[i3] = point.x;
    posArray[i3 + 1] = point.y;
    posArray[i3 + 2] = point.z;

    colorsArray[i3] = colorBlue.r;
    colorsArray[i3 + 1] = colorBlue.g;
    colorsArray[i3 + 2] = colorBlue.b;

    sizeArray[i] = Math.random() * 2 + 1.5;

    currentPositions[i3] = point.x;
    currentPositions[i3 + 1] = point.y;
    currentPositions[i3 + 2] = point.z;
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colorsArray, 3));
  geometry.setAttribute('size', new THREE.BufferAttribute(sizeArray, 1));

  const material = new THREE.ShaderMaterial({
    uniforms: { time: { value: 0 } },
    vertexShader: `
      attribute float size;
      attribute vec3 color;
      varying vec3 vColor;

      void main() {
        vColor = color;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * (300.0 / -mvPosition.z);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;

      void main() {
        float dist = distance(gl_PointCoord, vec2(0.5));
        if (dist > 0.5) discard;
        float glow = 1.0 - (dist * 2.0);
        gl_FragColor = vec4(vColor, glow);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  particles = new THREE.Points(geometry, material);
  scene.add(particles);

  const lineCount = (maxTokens * (maxTokens - 1)) / 2;
  attnLinesGeom = new THREE.BufferGeometry();
  const linePositions = new Float32Array(lineCount * 6);
  let lineIndex = 0;

  for (let i = 0; i < maxTokens; i += 1) {
    for (let j = i + 1; j < maxTokens; j += 1) {
      const angleA = (i / maxTokens) * Math.PI * 2;
      const angleB = (j / maxTokens) * Math.PI * 2;
      linePositions[lineIndex++] = Math.cos(angleA) * radius;
      linePositions[lineIndex++] = Math.sin(angleA) * radius;
      linePositions[lineIndex++] = -800;
      linePositions[lineIndex++] = Math.cos(angleB) * radius;
      linePositions[lineIndex++] = Math.sin(angleB) * radius;
      linePositions[lineIndex++] = -800;
    }
  }

  attnLinesGeom.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
  const lineMaterial = new THREE.LineBasicMaterial({
    color: 0x9d00ff,
    transparent: true,
    opacity: 0,
    blending: THREE.AdditiveBlending,
  });
  attnLinesMesh = new THREE.LineSegments(attnLinesGeom, lineMaterial);
  scene.add(attnLinesMesh);

  stateInfo.progress = 0;
  return true;
}

const ffnGeom = new THREE.BufferGeometry();
const ffnCount = 300;
const ffnPos = new Float32Array(ffnCount * 3);

for (let i = 0; i < ffnCount * 3; i += 1) {
  ffnPos[i] = (Math.random() - 0.5) * 1200;
  if (i % 3 === 2) {
    ffnPos[i] -= 1800;
  }
}

ffnGeom.setAttribute('position', new THREE.BufferAttribute(ffnPos, 3));
const ffnMat = new THREE.PointsMaterial({ color: 0x00ff66, size: 2, transparent: true, opacity: 0.1 });
const ffnMesh = new THREE.Points(ffnGeom, ffnMat);
scene.add(ffnMesh);

const ffnLinesGeom = new THREE.BufferGeometry();
const ffnLinePos: number[] = [];

for (let i = 0; i < ffnCount; i += 1) {
  for (let j = i + 1; j < ffnCount; j += 1) {
    const dx = ffnPos[i * 3] - ffnPos[j * 3];
    const dy = ffnPos[i * 3 + 1] - ffnPos[j * 3 + 1];
    const dz = ffnPos[i * 3 + 2] - ffnPos[j * 3 + 2];

    if (dx * dx + dy * dy + dz * dz < 40000) {
      ffnLinePos.push(ffnPos[i * 3], ffnPos[i * 3 + 1], ffnPos[i * 3 + 2]);
      ffnLinePos.push(ffnPos[j * 3], ffnPos[j * 3 + 1], ffnPos[j * 3 + 2]);
    }
  }
}

ffnLinesGeom.setAttribute('position', new THREE.Float32BufferAttribute(ffnLinePos, 3));
const ffnLineMat = new THREE.LineBasicMaterial({
  color: 0x0066ff,
  transparent: true,
  opacity: 0.05,
  blending: THREE.AdditiveBlending,
});
const ffnLinesMesh = new THREE.LineSegments(ffnLinesGeom, ffnLineMat);
scene.add(ffnLinesMesh);

async function fetchMiniMaxResponse(question: string) {
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        model: 'MiniMax-M2.7',
        max_tokens: 1024,
        messages: [{ role: 'user', content: question }],
      }),
    });

    const data = (await response.json()) as ApiResponse & { error?: string };

    if (!response.ok) {
      throw new Error(data.error ?? `HTTP Error: ${response.status}`);
    }

    const textBlock = data.content?.find((block) => block.type === 'text');
    return textBlock?.text ?? '接口返回成功，但没有文本内容。';
  } catch (error) {
    console.error(error);
    return `请求失败，链路中断：${error instanceof Error ? error.message : '未知错误'}`;
  }
}

function showStageDisplay(title: string, subTitle: string) {
  stageBigText.innerText = title;
  stageSubText.innerText = subTitle;
  stageOverlay.classList.add('visible');

  if (stageOverlayTimeout !== null) {
    window.clearTimeout(stageOverlayTimeout);
  }

  stageOverlayTimeout = window.setTimeout(() => {
    stageOverlay.classList.remove('visible');
    stageOverlayTimeout = null;
  }, 2500);
}

function addLog(message: string, highlight = false) {
  const div = document.createElement('div');
  div.className = `log-line ${highlight ? 'highlight' : ''}`;
  div.innerText = `> ${message}`;
  logContainer.appendChild(div);

  if (logContainer.children.length > 8) {
    logContainer.removeChild(logContainer.children[0]);
  }

  logContainer.scrollTop = logContainer.scrollHeight;
}

function startTypewriter(text: string) {
  aiResponseContainer.classList.remove('hidden');
  aiResponseText.innerText = '';

  let index = 0;
  if (typingInterval !== null) {
    window.clearInterval(typingInterval);
  }

  typingInterval = window.setInterval(() => {
    if (index < text.length) {
      aiResponseText.innerText += text.charAt(index);
      index += 1;
      return;
    }

    if (typingInterval !== null) {
      window.clearInterval(typingInterval);
      typingInterval = null;
    }
  }, 50);
}

function maybeStartTyping() {
  if (!isResponseReady || hasTriggeredTyping || stateInfo.progress < 0.88) {
    return;
  }

  hasTriggeredTyping = true;
  startTypewriter(globalAiResponse);
}

function setUiBusy(isBusy: boolean) {
  submitBtn.innerText = isBusy ? '数据演示中...' : '重新计算';
  submitBtn.disabled = isBusy;
  inputElement.disabled = isBusy;
}

function resetPanels() {
  attentionCalcPanel.classList.add('hidden');
  ffnCalcPanel.classList.add('hidden');
}

function runSimulation() {
  activeTimeline?.kill();
  activeCameraTimeline?.kill();

  stateInfo.progress = 0;
  camera.position.set(0, 0, 800);
  controls.target.set(0, 0, 0);
  controls.update();
  hasTriggeredTyping = false;

  activeTimeline = gsap.timeline();
  activeCameraTimeline = gsap.timeline();

  activeTimeline.to(stateInfo, {
    progress: 1,
    duration: 26,
    ease: 'none',
    onStart: () => {
      setUiBusy(true);
    },
    onUpdate: () => {
      const progress = stateInfo.progress;

      if (progress < 0.01 && hudStage.innerText !== '01 / Token 分词') {
        hudStage.innerText = '01 / Token 分词';
        hudDim.innerText = '文本维度';
        hudParams.innerText = `激活块数 ${maxTokens}`;
        addLog('正在将输入文本拆分为 Token 组块...', true);
        showStageDisplay('第一步: Token 词元化', `检测到 ${maxTokens} 个词元单元`);
        if (attnLinesMesh) {
          (attnLinesMesh.material as THREE.LineBasicMaterial).opacity = 0;
        }
      } else if (progress >= 0.1 && progress < 0.11 && hudStage.innerText !== '02 / Embedding 向量化') {
        hudStage.innerText = '02 / Embedding 向量化';
        hudDim.innerText = '4096 维';
        hudParams.innerText = '生成特征列';
        addLog('Token 已完成结构化，开始映射到高维特征空间。');
        showStageDisplay('第二步: Embedding 升维', '展示数据在高维空间中的稠密展开形态');
      } else if (progress >= 0.3 && progress < 0.31 && hudStage.innerText !== '03 / Attention 注意力') {
        hudStage.innerText = '03 / Attention 注意力';
        hudDim.innerText = '8192 窗口';
        hudParams.innerText = '计算上下文权重';
        addLog('启动自注意力交叉分析，开始计算词与词之间的上下文关联...', true);
        showStageDisplay('第三步: Self-Attention 交叉分析', '正在拆解 QKV 矩阵并计算注意力相似度');
        attentionCalcPanel.classList.remove('hidden');
      } else if (progress >= 0.6 && progress < 0.61 && hudStage.innerText !== '04 / FFN 前馈网络') {
        hudStage.innerText = '04 / FFN 前馈网络';
        hudDim.innerText = '混合形态';
        hudParams.innerText = '70.2B 突触';
        addLog('数据进入多层感知机丛林，完成特征抽取。');
        showStageDisplay('第四步: 多层感知机 (FFN)', '高维向量经过非线性激活层进行运算');
        if (attnLinesMesh) {
          (attnLinesMesh.material as THREE.LineBasicMaterial).opacity = 0;
        }
        attentionCalcPanel.classList.add('hidden');
        ffnCalcPanel.classList.remove('hidden');
      } else if (progress >= 0.85 && progress < 0.86 && hudStage.innerText !== '05 / 解码输出') {
        hudStage.innerText = '05 / 解码输出';
        hudDim.innerText = '概率聚合';
        addLog('运算结束，正在提取最高概率分布并组织回复。', true);
        showStageDisplay('第五步: 概率坍缩组装', '准备呈现 API 实时返回的回答');
        ffnCalcPanel.classList.add('hidden');
      }

      maybeStartTyping();

      if (progress >= 0.32 && progress <= 0.6) {
        const step = Math.floor((progress - 0.32) * 100);
        let text = "【拆分三组映射矩阵】\n特征 -> <span class='hlt-q'>[Q] 查询</span> / <span class='hlt-k'>[K] 键</span> / <span class='hlt-v'>[V] 值</span>\n\n";
        if (step > 2) text += '正在为目标 Token 匹配上下文候选...\n';
        if (step > 5) text += `-> <span class='hlt-q'>Q_target</span> · <span class='hlt-k'>K_context_1</span> = 0.${80 + (step % 19)} (高度相关)\n`;
        if (step > 8) text += `-> <span class='hlt-q'>Q_target</span> · <span class='hlt-k'>K_context_2</span> = 0.${10 + (step % 15)} (弱相关)\n`;
        if (step > 12) text += '\n【执行 Softmax 缩放】\n权重分布: [88%, 12%]\n';
        if (step > 15) text += "\n【融合目标结果】\n<span class='hlt-v'>V_new</span> = 权重 * <span class='hlt-v'>V_i</span>\n正在拉取数据并合并...";
        qkvLog.innerHTML = text;
      }

      if (progress >= 0.61 && progress <= 0.85) {
        const step = Math.floor((progress - 0.61) * 100);
        let text = "【非线性特征激活】\n参数映射 -> <span class='hlt-ffn'>MLP Layers</span>\n\n";
        if (step > 2) text += '正在计算特征残差连接...\n';
        if (step > 4) text += `-> ReLU( <span class='hlt-ffn'>xW1 + b1</span> ) 激活节点 #${3021 + step * 11}\n`;
        if (step > 7) text += '-> 映射维度扩散: 4096 -> 11008\n';
        if (step > 10) text += '-> 隐层感知机正在进行信息重组压缩...\n';
        if (step > 14) text += '\n【前馈输出完成】\n合并前向传播结果并执行 LayerNorm';
        ffnLog.innerHTML = text;
      }

      if (progress >= 0.45 && progress <= 0.6 && attnLinesMesh) {
        (attnLinesMesh.material as THREE.LineBasicMaterial).opacity = 0.5 + Math.sin(progress * 50) * 0.3;
      } else if (progress > 0.6) {
        attnLinesMesh && ((attnLinesMesh.material as THREE.LineBasicMaterial).opacity = 0);
      }

      if (progress >= 0.6 && progress <= 0.85) {
        const ffnProgress = (progress - 0.6) / 0.25;
        ffnMat.opacity = 0.3 + ffnProgress * 0.5 + Math.random() * 0.2;
        ffnLineMat.opacity = 0.1 + ffnProgress * 0.2 + Math.random() * 0.1;
      } else {
        ffnMat.opacity = 0.1;
        ffnLineMat.opacity = 0.05;
      }
    },
    onComplete: () => {
      setUiBusy(false);
      inputElement.value = '';
      addLog('系统就绪，等待下一次对话。');
    },
  });

  activeCameraTimeline
    .to(camera.position, { z: 400, y: 0, duration: 4 })
    .to(camera.position, { z: 100, x: -100, y: 100, duration: 6, ease: 'power1.inOut' })
    .to(camera.position, { z: -400, x: 0, y: 0, duration: 8, ease: 'sine.inOut' })
    .to(camera.position, { z: -1400, x: 0, y: 0, duration: 5, ease: 'power2.inOut' })
    .to(camera.position, { z: -2700, x: 0, y: 0, duration: 3, ease: 'power1.out' });
}

submitBtn.addEventListener('click', async () => {
  const value = inputElement.value.trim();
  if (!value) {
    return;
  }

  aiResponseContainer.classList.add('hidden');
  resetPanels();

  globalAiResponse = '等待模型返回中...';
  isResponseReady = false;
  updateTargetOutput('正在解码');

  const initialized = initOrUpdateParticles(value);
  if (!initialized) {
    addLog('输入内容没有生成可渲染的字符轮廓，请换一个问题再试。', true);
    return;
  }

  runSimulation();

  const responseText = await fetchMiniMaxResponse(value);
  globalAiResponse = responseText;
  isResponseReady = true;

  const cleanText = responseText.replace(/[\r\n\s]+/g, ' ');
  let shortText = cleanText.substring(0, 10);
  if (cleanText.length > 10) {
    shortText += '...';
  }
  updateTargetOutput(shortText || '输出完成');
  maybeStartTyping();
});

inputElement.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    submitBtn.click();
  }
});

renderer.domElement.addEventListener('mousemove', (event) => {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
});

renderer.domElement.addEventListener('mouseleave', () => {
  scatterFactor = 0;
});

addLog('通过服务端代理连接 MiniMax 接口。');
initOrUpdateParticles('大模型全息接口');

function updateParticles() {
  if (!geometry || !particles || pCount === 0) {
    return;
  }

  raycaster.setFromCamera(mouse, camera);
  const rayOrigin = raycaster.ray.origin;
  const rayDir = raycaster.ray.direction;
  const planeZ = 0;
  if (Math.abs(rayDir.z) > 0.001) {
    const t = (planeZ - rayOrigin.z) / rayDir.z;
    if (t > 0) {
      mouseWorld.x = rayOrigin.x + rayDir.x * t;
      mouseWorld.y = rayOrigin.y + rayDir.y * t;
    }
  }
  mouseWorld.z = 0;

  const progress = stateInfo.progress;
  const posAttr = geometry.attributes.position as THREE.BufferAttribute;
  const colAttr = geometry.attributes.color as THREE.BufferAttribute;
  const now = Date.now() * 0.005;

  const minX = Math.min(...posTokens.map(p => p.x));
  const maxX = Math.max(...posTokens.map(p => p.x));
  const minY = Math.min(...posTokens.map(p => p.y));
  const maxY = Math.max(...posTokens.map(p => p.y));

  if (progress < 0.1 && mouseWorld.x > minX - 50 && mouseWorld.x < maxX + 50 && mouseWorld.y > minY - 50 && mouseWorld.y < maxY + 50) {
    scatterFactor = Math.min(scatterFactor + 0.08, 1);
  } else {
    scatterFactor *= 0.92;
  }

  for (let i = 0; i < pCount; i += 1) {
    const i3 = i * 3;

    let sourcePoint: THREE.Vector3 | null = null;
    let targetPoint: THREE.Vector3 | null = null;
    let t = 0;
    let targetColor = colorBlue;

    if (progress < 0.1) {
      const base = posTokens[i];
      if (scatterFactor > 0.1) {
        const dx = base.x - mouseWorld.x;
        const dy = base.y - mouseWorld.y;
        const dist = Math.sqrt(dx * dx + dy * dy) + 1;
        const repel = (300 / dist) * scatterFactor;
        const angle = Math.atan2(dy, dx);
        const wobble = Math.sin(now * 3 + i * 0.5) * 20 * scatterFactor;
        const wobble2 = Math.cos(now * 2.5 + i * 0.3) * 12 * scatterFactor;
        currentPositions[i3] = base.x + Math.cos(angle) * repel + Math.cos(angle + Math.PI / 2) * wobble;
        currentPositions[i3 + 1] = base.y + Math.sin(angle) * repel + Math.sin(angle + Math.PI / 2) * wobble;
        currentPositions[i3 + 2] = base.z + wobble2 + Math.sin(now * 4 + i) * 10 * scatterFactor;
      } else {
        currentPositions[i3] = base.x;
        currentPositions[i3 + 1] = base.y;
        currentPositions[i3 + 2] = base.z;
      }
    } else if (progress < 0.15) {
      sourcePoint = posTokens[i];
      targetPoint = posEmbeds[i];
      t = (progress - 0.1) / 0.05;
    } else if (progress < 0.3) {
      currentPositions[i3] = posEmbeds[i].x;
      currentPositions[i3 + 1] = posEmbeds[i].y;
      currentPositions[i3 + 2] = posEmbeds[i].z;
    } else if (progress < 0.35) {
      sourcePoint = posEmbeds[i];
      targetPoint = posAttn[i];
      t = (progress - 0.3) / 0.05;
      targetColor = colorPurple;
    } else if (progress < 0.6) {
      currentPositions[i3] = posAttn[i].x;
      currentPositions[i3 + 1] = posAttn[i].y + Math.sin(now + i) * 10;
      currentPositions[i3 + 2] = posAttn[i].z;

      const qkvKind = i % 3;
      targetColor = qkvKind === 0 ? colorQuery : qkvKind === 1 ? colorKey : colorValue;
    } else if (progress < 0.65) {
      sourcePoint = posAttn[i];
      targetPoint = posFFN[i];
      t = (progress - 0.6) / 0.05;
      targetColor = colorGreen;
    } else if (progress < 0.8) {
      const burst = Math.sin(now * 5 + i) > 0.8 ? 150 : 0;
      currentPositions[i3] = posFFN[i].x + Math.sin(now + i) * (20 + burst);
      currentPositions[i3 + 1] = posFFN[i].y + Math.cos(now + i) * (20 + burst);
      currentPositions[i3 + 2] = posFFN[i].z + Math.sin(now * 2 - i) * (20 + burst) + (Math.random() - 0.5) * burst;
      targetColor = burst > 0 ? colorBurst : colorGreen;
    } else if (progress < 0.85) {
      sourcePoint = posFFN[i];
      targetPoint = posOut[i];
      t = (progress - 0.8) / 0.05;
    } else {
      const point = posOut[i];
      const easeOut = (progress - 0.85) / 0.15;
      const jitter = (1 - easeOut) * 20;
      currentPositions[i3] = point.x + (Math.random() - 0.5) * jitter;
      currentPositions[i3 + 1] = point.y + (Math.random() - 0.5) * jitter;
      currentPositions[i3 + 2] = point.z + (Math.random() - 0.5) * jitter;
    }

    if (sourcePoint && targetPoint) {
      const smoothT = t * t * (3 - 2 * t);
      currentPositions[i3] = sourcePoint.x + (targetPoint.x - sourcePoint.x) * smoothT;
      currentPositions[i3 + 1] = sourcePoint.y + (targetPoint.y - sourcePoint.y) * smoothT;
      currentPositions[i3 + 2] = sourcePoint.z + (targetPoint.z - sourcePoint.z) * smoothT;
    }

    const origR = colAttr.array[i3];
    const origG = colAttr.array[i3 + 1];
    const origB = colAttr.array[i3 + 2];
    const mix = 0.1;
    colAttr.array[i3] = origR + (targetColor.r - origR) * mix;
    colAttr.array[i3 + 1] = origG + (targetColor.g - origG) * mix;
    colAttr.array[i3 + 2] = origB + (targetColor.b - origB) * mix;
  }

  posAttr.copyArray(currentPositions);
  posAttr.needsUpdate = true;
  colAttr.needsUpdate = true;

  if (attnLinesMesh) {
    attnLinesMesh.rotation.z += 0.002;
  }
  ffnMesh.rotation.y += 0.001;
  ffnLinesMesh.rotation.y += 0.001;
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  updateParticles();
  cameraLookTarget.set(0, 0, camera.position.z - 400);
  camera.lookAt(cameraLookTarget);
  composer.render();
}

animate();
