"""Audit published navigation, including anchors and files absent from Git."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import ssl
import subprocess
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

import certifi

ROOT=Path(__file__).resolve().parents[2]
TLS=ssl.create_default_context()
TLS.load_verify_locations(certifi.where())
GIT='git'


class HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[];self.ids=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        for key in ['href','src']:
            if key in attrs:self.links.append(attrs[key])
        if 'id' in attrs:self.ids.append(attrs['id'])


def markdown(text):
    links=re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)',text)
    links+=re.findall(r'<(https?://[^>]+)>',text)
    counts={};ids=[]
    for title in re.findall(r'^#{1,6}\s+(.+?)\s*#*\s*$',text,re.M):
        slug=re.sub(r'[^\w\s-]','',title.lower()).strip().replace(' ','-')
        index=counts.get(slug,0);counts[slug]=index+1
        ids.append(slug+('-'+str(index) if index else ''))
    ids+=re.findall(r'\bid=[\'"]([^\'"]+)',text)
    return links,ids


def parse(path):
    text=path.read_text('utf-8-sig')
    if path.suffix=='.ipynb':
        text='\n'.join(''.join(c['source'])for c in json.loads(text)['cells']if c['cell_type']=='markdown')
    if path.suffix=='.html':
        parser=HTMLLinks();parser.feed(text);return parser.links,parser.ids
    return markdown(text)


def external(url):
    for method in ['HEAD','GET']:
        try:
            request=Request(url,method=method,headers={'User-Agent':'PoetryGPT-link-audit/1.0','Range':'bytes=0-511'})
            with urlopen(request,context=TLS,timeout=25)as response:
                if method=='GET':response.read(512)
                return dict(url=url,status=response.status,final_url=response.url.split('?')[0])
        except HTTPError as error:
            result=dict(url=url,status=error.code,error=error.reason)
            if method=='HEAD' and error.code in [403,405,429,502,503]:continue
            return result
        except Exception as error:
            result=dict(url=url,status=None,error=str(error))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external',action='store_true',help='Also check external navigation and the inference archive URL')
    args=parser.parse_args()
    files=set(subprocess.check_output([GIT,'-C',str(ROOT),'ls-files'],text=True).splitlines())
    documents={name:parse(ROOT/name)for name in sorted(files)
        if Path(name).suffix in ['.md','.html','.ipynb'] and not name.endswith('report_template.html')}
    broken=[];external_urls={};count=0
    for name,(links,_)in documents.items():
        for url in links:
            count+=1;parsed=urlsplit(url)
            github_target=None
            if parsed.scheme in ['http','https']:
                external_urls.setdefault(url,[]).append(name)
                for kind in ['blob','tree']:
                    prefix=f'https://github.com/Buyan-Kirill/NLP_course/{kind}/main/'
                    if url.startswith(prefix):
                        github_target=ROOT/unquote(parsed.path.split('/main/',1)[1])
                        break
                if github_target is None:
                    continue
            elif parsed.scheme:
                broken.append(dict(source=name,url=url,reason='nonportable URL scheme'))
                continue
            target=github_target.resolve() if github_target is not None else (
                ((ROOT/name).parent/unquote(parsed.path)).resolve() if parsed.path else ROOT/name)
            try:relative=target.relative_to(ROOT).as_posix()
            except ValueError:broken.append(dict(source=name,url=url,reason='outside repo'));continue
            if relative not in files and not any(f.startswith(relative.rstrip('/')+'/') for f in files):
                broken.append(dict(source=name,url=url,reason='not in Git'));continue
            if parsed.fragment and relative in documents:
                if unquote(parsed.fragment)not in documents[relative][1]:
                    broken.append(dict(source=name,url=url,reason='anchor missing'))
    resources=json.loads((ROOT/'hw01/reproduction/resources.json').read_text('utf8'))
    external_urls.setdefault(resources['inference']['url'],[]).append('hw01/reproduction/resources.json')
    checks=[]
    if args.external:
        with ThreadPoolExecutor(max_workers=4)as pool:
            checks=list(pool.map(external,external_urls))
    result=dict(documents=len(documents),navigation_links=count,local_broken=broken,
        external=[dict(**r,sources=external_urls[r['url']])for r in checks])
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if broken or any(row['status'] is None or row['status']>=400 for row in checks):
        raise SystemExit(1)


if __name__=='__main__':main()
