import { H, TICK, loadBeam, signature } from './world.mjs?v=2';
import { gpu } from './gpu.mjs?v=2';
import { connect, lockstep } from './peer.mjs?v=2';
import { view } from './view.mjs?v=2';

const el = id => document.getElementById(id);
const controls = ['fire-a', 'fire-b'].map(el);
const dialog = el('link');
let w, engine, snapshot, draw, connection, session, connectionEpoch = 0;
let queued = [0, 0], deadline = 0, busy = false, blocked = false;
const period = 1000 / TICK;
const notify = text => { if (el('peer-status').textContent !== text) el('peer-status').textContent = text; };

function refresh() {
  if (!w || !snapshot) return;
  const done = w.finished(snapshot), lost = blocked || !!session?.fault;
  controls.forEach((b, i) => {
    const allowed = !session || session.side === i;
    b.disabled = lost || !allowed || done || !!snapshot[H.broken] || !!(snapshot[H.fired] & (1 << i));
  });
  el('reset').disabled = lost;
  const text = session?.fault || (snapshot[H.fault] ? 'Переполнение состояния'
    : snapshot[H.separated] ? 'Пролёт отделён'
    : done ? 'Пролёт выдержал'
    : snapshot[H.fired] === 0 ? 'Готов к испытанию'
    : snapshot[H.fired] === 3 ? 'Ответ конструкции'
    : 'Пролёт раскачивается');
  if (el('result').textContent !== text) el('result').textContent = text;
  const time = (snapshot[H.age] / w.asset.hz).toFixed(2).replace('.', ',') + ' с';
  if (el('time').textContent !== time) el('time').textContent = time;
  let sync = 'Одиночная игра';
  if (session) {
    sync = session.fault || (blocked ? 'P2P · пауза' : 'P2P · игрок ' + (session.side === 0 ? 'A' : 'B'));
    if (!lost && session.agreed) sync += ' · ' + session.agreed;
  }
  if (el('sync').textContent !== sync) el('sync').textContent = sync;
}

function action(command, player) {
  if (blocked || !engine || (command === 1 && controls[player].disabled)) return;
  if (session) session.queue(command);
  else queued[player] = Math.max(command, queued[player]);
}
controls.forEach((button, i) => button.addEventListener('click', () => action(1, i)));
el('reset').onclick = () => action(2, session?.side ?? 0);
document.addEventListener('keydown', event => {
  if (event.repeat || dialog.open || /^(INPUT|TEXTAREA)$/.test(event.target.tagName)) return;
  if (event.key === '1' || event.key === '2') action(1, Number(event.key) - 1);
  if (event.code === 'KeyR') action(2, session?.side ?? 0);
});

async function frame(now) {
  requestAnimationFrame(frame);
  if (!engine || busy || blocked || now < deadline) return;
  busy = true;
  try {
    if (now - deadline > 100) deadline = now;
    for (let i = 0; i < 3 && now >= deadline; i++) {
      const activeSession = session;
      const packet = activeSession ? activeSession.advance() : { ops: queued.splice(0, 2, 0, 0) };
      if (!packet) {
        deadline = now + period / 2;
        break;
      }
      deadline += period;
      const { ops, tick } = packet;
      const mask = (ops[0] === 1 ? 1 : 0) | (ops[1] === 1 ? 2 : 0);
      if (mask || ops.includes(2) || (snapshot[H.fired] && !w.finished(snapshot))) {
        snapshot = await engine.step(mask, ops.includes(2));
      }
      activeSession?.checkpoint(tick, signature(snapshot));
      if (blocked) break;
      if (snapshot[H.fault]) { blocked = true; break; }
    }
    refresh();
  } catch (error) {
    blocked = true;
    el('result').textContent = error.message;
  } finally {
    busy = false;
  }
}

let lastFrame = '';
function animation() {
  if (snapshot) {
    const canvas = el('scene'), side = session?.side ?? 0;
    const key = [snapshot[H.age], snapshot[H.fired], side, canvas.clientWidth, canvas.clientHeight, devicePixelRatio].join('/');
    if (key !== lastFrame) {
      draw(snapshot, side);
      lastFrame = key;
    }
  }
  requestAnimationFrame(animation);
}

function ice() {
  const urls = el('ice').value.trim();
  if (!urls) return [];
  if (!/^(stun|turn|turns):/.test(urls)) throw new Error('Укажи адрес STUN или TURN');
  return [{ urls, username: el('login').value, credential: el('password').value }];
}

function clearLink() {
  connectionEpoch++;
  connection?.close();
  connection = undefined;
  session = undefined;
  blocked = false;
  deadline = 0;
  el('outgoing').value = '';
  el('apply').disabled = true;
  el('copy').disabled = true;
  el('offline').disabled = true;
}

async function prepare(side) {
  if (!engine) return;
  clearLink();
  const epoch = connectionEpoch;
  try {
    notify('Собираем код подключения…');
    blocked = true;
    refresh();
    el('offline').disabled = false;
    connection = connect(side, ice(), async channel => {
      if (epoch !== connectionEpoch) return;
      const a = w.asset;
      const version = 'sb2-' + signature([a.rotation.mode, a.bits, a.hz, a.contact_stiffness,
        a.seam_stiffness, a.striker_mass, a.speed, ...a.modes.flat(), ...a.bonds.flat(),
        ...a.ports.flat(), ...a.limits, ...a.launch]);
      // Receive the hello immediately, including while the local engine resets.
      session = lockstep(channel, side, version);
      try {
        while (busy) await new Promise(resolve => requestAnimationFrame(resolve));
        if (epoch !== connectionEpoch) return;
        snapshot = await engine.step(0, true);
        if (epoch !== connectionEpoch) return;
        queued = [0, 0];
        deadline = 0;
        blocked = false;
        dialog.close();
        notify('Подключено');
        refresh();
      } catch (error) { notify(error.message); }
    }, message => {
      if (epoch !== connectionEpoch) return;
      blocked = true;
      notify(message);
      refresh();
    });
    const token = await connection.prepare(el('incoming').value);
    if (epoch !== connectionEpoch) return;
    el('outgoing').value = token;
    el('copy').disabled = false;
    el('apply').disabled = side !== 0;
    notify(side === 0 ? 'Отправь свой код игроку B. Затем вставь его ответ и примени.'
      : 'Отправь свой код игроку A и дождись подключения.');
  } catch (error) {
    if (epoch !== connectionEpoch) return;
    clearLink();
    notify(error.message);
    refresh();
  }
}

el('invite').onclick = () => dialog.showModal();
el('host').onclick = () => prepare(0);
el('join').onclick = () => prepare(1);
el('apply').onclick = async () => {
  try {
    await connection.accept(el('incoming').value);
    el('apply').disabled = true;
    notify('Устанавливаем соединение…');
  } catch (error) { notify(error.message); }
};
el('copy').onclick = async () => {
  try { await navigator.clipboard.writeText(el('outgoing').value); }
  catch { el('outgoing').select(); }
};
el('offline').onclick = async () => {
  clearLink();
  blocked = true;
  while (busy) await new Promise(resolve => requestAnimationFrame(resolve));
  snapshot = await engine.step(0, true);
  queued = [0, 0];
  blocked = false;
  dialog.close();
  refresh();
};

async function start() {
  w = await loadBeam();
  let label = 'JS · целочисленный эталон', device;
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter && !adapter.info.isFallbackAdapter) {
        device = await adapter.requestDevice();
        engine = await gpu(w, device);
        label = 'WebGPU' + (adapter.info.vendor ? ' · ' + adapter.info.vendor : '');
        device.lost.then(info => {
          if (info.reason === 'destroyed') return;
          blocked = true;
          el('result').textContent = 'Устройство GPU потеряно. Обнови страницу.';
          controls.forEach(button => button.disabled = true);
          el('reset').disabled = true;
        });
      }
    } catch (error) {
      device?.destroy();
      console.warn('WebGPU initialization failed', error);
    }
  }
  engine ||= w.cpu();
  snapshot = w.state.slice();
  draw = view(el('scene'), w);
  el('backend').textContent = label;
  refresh();
  requestAnimationFrame(animation);
  requestAnimationFrame(frame);
}

start().catch(error => { el('result').textContent = error.message; });
