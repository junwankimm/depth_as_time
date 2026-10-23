"""Inline data.json, toy.json and the EN/KO page bodies into template.html -> index.html."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(HERE, 'template.html')).read()
for key, f in [('__DATA__', 'data.json'), ('__TOY__', 'toy.json')]:
    blob = open(os.path.join(HERE, f)).read()
    assert '</' not in blob
    s = s.replace(key, blob)
ids = {}
for key, f in [('__BODY_EN__', 'body_en.html'), ('__BODY_KO__', 'body_ko.html')]:
    body = open(os.path.join(HERE, f)).read()
    assert '</template' not in body and '<script' not in body
    import re
    ids[f] = re.findall(r'\bid="([^"]+)"', body)
    s = s.replace(key, body)
# both languages must expose the same element ids, in the same order
assert ids['body_en.html'] == ids['body_ko.html'], set(ids['body_en.html']) ^ set(ids['body_ko.html'])
open(os.path.join(HERE, 'index.html'), 'w').write(s)
print(len(s))
