"""Run index.html's script in QuickJS against a minimal DOM shim.

Catches syntax errors (including duplicate const/let) and runtime errors in the
initial render, then fires every registered input/click listener once.
Runs once per language (en, ko).
Usage: PYTHONPATH=<dir with quickjs> python check_page.py [index.html]
"""
import re
import sys

import quickjs

SHIM = r"""
var __listeners = [], __els = {};
function FakeEl(id) {
  this.id = id; this.children = []; this.style = {}; this.dataset = {}; this.attrs = {};
  this.innerHTML = ''; this.textContent = ''; this.value = '0'; this.hidden = false;
  this.offsetWidth = 100; this.className = '';
}
FakeEl.prototype.setAttribute = function (k, v) { this.attrs[k] = String(v); };
FakeEl.prototype.getAttribute = function (k) { return this.attrs[k]; };
FakeEl.prototype.appendChild = function (c) { this.children.push(c); return c; };
FakeEl.prototype.addEventListener = function (ev, fn) { __listeners.push([this, ev, fn]); };
FakeEl.prototype.getBoundingClientRect = function () { return {left: 0, top: 0, width: 460, height: 260}; };
FakeEl.prototype.querySelectorAll = function (sel) { return document.querySelectorAll(sel); };
FakeEl.prototype.querySelector = function (sel) { return new FakeEl(sel); };
FakeEl.prototype.closest = function () { return this; };
var document = {
  getElementById: function (id) {
    if (id === 'data') return {textContent: __DATA};
    if (id === 'toydata') return {textContent: __TOY};
    if (!__els[id]) __els[id] = new FakeEl(id);
    return __els[id];
  },
  querySelector: function (sel) { return new FakeEl(sel); },
  createElementNS: function (ns, tag) { return new FakeEl(tag); },
  createElement: function (tag) { return new FakeEl(tag); },
  querySelectorAll: function (sel) {
    var out = [];
    for (var k in __els) if (sel.indexOf('.') === 0 || sel.indexOf('#') >= 0) out.push(__els[k]);
    return out;
  },
  documentElement: new FakeEl('html'), body: new FakeEl('body')
};
var window = {matchMedia: function () { return {matches: false, addEventListener: function () {}}; },
              addEventListener: function () {}};
var localStorage = {getItem: function () { return __LANG; }, setItem: function () {}};
function setInterval() { return 1; } function clearInterval() {} function setTimeout(f) { return 1; }
function requestAnimationFrame() { return 1; }
var console = {log: function () {}, warn: function () {}, error: function () {}};
"""


def main(path, lang='en'):
    s = open(path).read()
    data = re.search(r'<script id="data" type="application/json">(.*?)</script>', s, re.S).group(1)
    toy_m = re.search(r'<script id="toydata" type="application/json">(.*?)</script>', s, re.S)
    toy = toy_m.group(1) if toy_m else '{}'
    js = re.findall(r'<script>(.*?)</script>', s, re.S)[-1]
    ctx = quickjs.Context()
    ctx.set('__DATA', data)
    ctx.set('__TOY', toy)
    ctx.set('__LANG', lang)
    ctx.eval(SHIM)
    ctx.eval(js)
    print(f'[{lang}] initial render ok; listeners:', ctx.eval('__listeners.length'))
    n_err = ctx.eval(r"""
      var errs = [];
      __listeners.forEach(function (l, i) {
        var tgt = l[0];
        var b = new FakeEl('btn'); var V = {toyDs: 'ring8', toyTeacher: 'flm', toyTau: '0.50', swapSeg: 'owt', depSeg: 'owt'};
        b.dataset = {i: '1', v: V[tgt.id] || '1', n: '4', l: tgt.id === 'langSeg' ? (__LANG === 'ko' ? 'en' : 'ko') : undefined};
        b.closest = function () { return b; };
        try { l[2]({target: l[1] === 'input' ? {value: '3'} : b, clientX: 50, clientY: 50}); }
        catch (e) { errs.push((tgt.id || '?') + ' ' + l[1] + ': ' + e + ' @ ' + (e.stack || '').split('\n').slice(0, 3).join(' / ')); }
      });
      globalThis.__errs = errs.join('\n');
      errs.length;
    """)
    if n_err:
        print('listener errors:\n' + ctx.eval('__errs'))
    print('listener check done, errors:', n_err)
    return n_err


if __name__ == '__main__':
    page = sys.argv[1] if len(sys.argv) > 1 else 'index.html'
    sys.exit(1 if sum(main(page, lang) for lang in ('en', 'ko')) else 0)
