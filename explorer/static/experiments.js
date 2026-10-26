// Shared by neuron chat and the tutorial; neither owns the other's conversation.
export async function streamSteering(payload, signal, onEvent) {
  const res = await fetch('/api/steer', {method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify(payload), signal});
  if (!res.ok) { const problem = await res.json(); throw new Error(problem.error || 'Could not run Qwen'); }
  const reader = res.body.getReader(), decoder = new TextDecoder();
  let buffer = '', finished = false;
  try {
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream:true});
      let end;
      while ((end = buffer.indexOf('\n\n')) >= 0) {
        const line = buffer.slice(0, end); buffer = buffer.slice(end + 2);
        if (!line.startsWith('data: ')) continue;
        const event = JSON.parse(line.slice(6));
        if (event.error) throw new Error(event.error);
        if (event.cancelled) throw new DOMException('Generation stopped', 'AbortError');
        if (event.row == null && !event.meta) finished = true;
        onEvent(event);
      }
    }
    if (!finished) throw new Error('Generation connection ended early. Retry the run.');
  } finally { reader.releaseLock(); }
}
export const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export async function readNeuron(neuron, text, signal) {
  const response = await fetch('/api/reads', {method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({...neuron,text,chat:true}),signal});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Activation run failed');
  return result;
}
export function activationHTML(result) {
  const maximum = Math.max(1e-6, ...result.units.map(u => Math.abs(u.act)));
  return result.units.map(u => {
    const value = u.act / maximum;
    const color = value >= 0 ? `rgba(255,181,71,${value*.85})` : `rgba(255,79,154,${-value*.85})`;
    return `<span tabindex="0" style="background:${color}" title="Activation ${u.act.toFixed(3)}; token pieces ${u.pieces.map(v => v.toFixed(3)).join(' · ')}">${escapeHTML(u.text)}</span>`;
  }).join('');
}
