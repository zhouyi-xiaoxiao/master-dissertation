#!/usr/bin/env python
"""Look up every DOI of refs.bib on Crossref and print the registered metadata
(check, made while the article was prepared, that each reference exists with the stated data).
Usage: python reference_checks.py [refs.bib]   -> ../data/references_crossref.json"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, re, sys, os, urllib.request, urllib.parse, time
HERE = os.path.dirname(os.path.abspath(__file__))
bib = sys.argv[1] if len(sys.argv) > 1 else _os.path.join(_R, 'refs.bib')
dois = sys.argv[2:] if len(sys.argv) > 2 else re.findall(r'doi\s*=\s*\{([^}]+)\}', open(bib).read())
out = {}
for d in dois:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(d)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "refcheck/1.0 (mailto:zhouyixiaoxiao@gmail.com)"})
        m = json.load(urllib.request.urlopen(req, timeout=30))["message"]
        au = "; ".join((a.get("family", "") + ", " + a.get("given", "")) for a in m.get("author", []))
        rec = {"title": (m.get("title") or [""])[0], "authors": au,
               "container": (m.get("container-title") or [""])[0], "volume": m.get("volume"),
               "issue": m.get("issue"), "page": m.get("page"), "article": m.get("article-number"),
               "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0], "type": m.get("type"),
               "publisher": m.get("publisher")}
    except Exception as e:
        rec = {"error": repr(e)}
    out[d] = rec
    print(d, "|", json.dumps(rec, ensure_ascii=False))
    time.sleep(0.3)
json.dump(out, open(_os.path.join(_R, 'data', 'article', 'references_crossref.json'), "w"), indent=1, ensure_ascii=False)
