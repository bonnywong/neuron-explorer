import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source=await readFile(new URL('../explorer/static/experiments.js',import.meta.url),'utf8');
const {streamSteering,activationHTML}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

function response(chunks) {
  return new Response(new ReadableStream({start(c) { for(const chunk of chunks)c.enqueue(new TextEncoder().encode(chunk));c.close(); }}));
}
test('steering assembles split SSE events and keeps actual metadata',async()=>{
  globalThis.fetch=async()=>response(['data: {"meta":{"device":"cuda"}}\n\n','data: {"row":0,"t":"actual',' output"}\n\ndata: {"row":null,"t":null}\n\n']);
  const seen=[]; await streamSteering({},undefined,e=>seen.push(e));
  assert.equal(seen[0].meta.device,'cuda'); assert.equal(seen[1].t,'actual output'); assert.equal(seen.length,3);
});
test('runtime cancellation cannot be presented as a completed experiment',async()=>{
  globalThis.fetch=async()=>response(['data: {"row":null,"cancelled":true}\n\n']);
  await assert.rejects(streamSteering({},undefined,()=>{}),{name:'AbortError'});
});
test('truncated stream is an error rather than a successful empty output',async()=>{
  globalThis.fetch=async()=>response(['data: {"row":0,"t":"partial"}\n\n']);
  await assert.rejects(streamSteering({},undefined,()=>{}),/ended early/);
});
test('structured capability errors reach the UI',async()=>{
  globalThis.fetch=async()=>Response.json({error:'Open model setup'},{status:409});
  await assert.rejects(streamSteering({},undefined,()=>{}),/Open model setup/);
});
test('activation display escapes text and exposes keyboard-readable values',()=>{
  const html=activationHTML({units:[{text:'<script>',act:-2,pieces:[-2]}]});
  assert.ok(html.includes('&lt;script&gt;')); assert.ok(html.includes('tabindex="0"')); assert.ok(html.includes('Activation -2.000'));
});
