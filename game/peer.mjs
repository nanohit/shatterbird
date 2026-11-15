const LEAD = 8;
const CAP = 256;

// Both peers supply each tick. Connection setup and the packet arrival order never enter physics.
export function lockstep(channel, side, version) {
  const mine = new Map(), theirs = new Map(), hashes = [new Map(), new Map()];
  let tick = 0, sent = 0, received = 0, pending = 0, hello = false, fault = '', agreed = '';
  const send = packet => channel.send(JSON.stringify(packet));
  channel.onmessage = event => {
    try {
      if (fault || typeof event.data !== 'string' || event.data.length > 128) throw new Error('Ошибка протокола');
      const packet = JSON.parse(event.data);
      if (!Array.isArray(packet) || packet.length !== 3) throw new Error('Ошибка протокола');
      const [kind, t, value] = packet;
      if (kind === 'hello') {
        if (hello || t !== 1 - side || value !== version) throw new Error('Версии игры или роли отличаются');
        hello = true;
      } else if (kind === 'input' && hello) {
        if (t !== received || t > tick + CAP || !Number.isInteger(value) || value < 0 || value > 2) {
          throw new Error('Неверный ввод партнёра');
        }
        theirs.set(t, value);
        received++;
      } else if (kind === 'hash' && hello) {
        if (!Number.isInteger(t) || t <= 0 || t % 60 || Math.abs(t - tick) > CAP ||
            typeof value !== 'string' || !/^[\da-f]{8}$/.test(value)) {
          throw new Error('Неверная отметка состояния');
        }
        check(1, t, value);
      } else throw new Error('Ошибка протокола');
    } catch (error) { fault ||= error.message; }
  };
  function check(sender, t, h) {
    const other = hashes[1 - sender];
    if (!other.has(t)) hashes[sender].set(t, h);
    else {
      if (other.get(t) !== h) fault = `Рассинхрон на тике ${t}`;
      else agreed = h;
      other.delete(t);
    }
  }
  send(['hello', side, version]);
  return {
    side,
    queue(command) { pending = Math.max(command, pending); },
    advance() {
      if (fault || !hello || channel.readyState !== 'open') return null;
      while (sent < tick + LEAD) {
        mine.set(sent, pending);
        send(['input', sent, pending]);
        pending = 0;
        sent++;
      }
      if (!theirs.has(tick)) return null;
      const ops = side === 0 ? [mine.get(tick), theirs.get(tick)] : [theirs.get(tick), mine.get(tick)];
      mine.delete(tick);
      theirs.delete(tick);
      tick++;
      return { tick, ops };
    },
    checkpoint(t, h) {
      if (t % 60 === 0 && !fault && channel.readyState === 'open') {
        check(0, t, h);
        send(['hash', t, h]);
      }
    },
    get fault() { return fault; },
    get agreed() { return agreed; },
  };
}

const encode = desc => btoa(JSON.stringify(desc));
function decode(token, type) {
  const obj = JSON.parse(atob(token.trim()));
  if (obj.type !== type || typeof obj.sdp !== 'string') throw new Error('Код подключения повреждён');
  return obj;
}

async function gather(peer) {
  if (peer.iceGatheringState === 'complete') return;
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      peer.removeEventListener('icegatheringstatechange', check);
      reject(new Error('Подключение не собрано. Проверь адрес STUN/TURN.'));
    }, 14000);
    function check() {
      if (peer.iceGatheringState !== 'complete') return;
      clearTimeout(timer);
      peer.removeEventListener('icegatheringstatechange', check);
      resolve();
    }
    peer.addEventListener('icegatheringstatechange', check);
    check();
  });
}

export function connect(side, iceServers, opened, status) {
  const relay = iceServers.some(({ urls }) => /^turns?:/.test(urls));
  const peer = new RTCPeerConnection({ iceServers, iceTransportPolicy: relay ? 'relay' : 'all' });
  function bind(channel) {
    channel.onopen = () => opened(channel);
    channel.onclose = () => status('Партнёр отключился');
  }
  if (side === 0) bind(peer.createDataChannel('second-impact'));
  else peer.ondatachannel = event => bind(event.channel);
  peer.onconnectionstatechange = () => {
    if (peer.connectionState === 'failed') status('Соединение не установлено. Можно попробовать TURN.');
  };
  return {
    async prepare(token) {
      if (side === 1) await peer.setRemoteDescription(decode(token, 'offer'));
      await peer.setLocalDescription(side === 0 ? await peer.createOffer() : await peer.createAnswer());
      await gather(peer);
      return encode(peer.localDescription);
    },
    async accept(token) { await peer.setRemoteDescription(decode(token, 'answer')); },
    close() { peer.close(); },
  };
}
