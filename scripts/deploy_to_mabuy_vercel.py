#!/usr/bin/env python3
import os, json, base64, urllib.request, urllib.error

VERCEL_TOKEN = os.environ.get('VERCEL_TOKEN', '')
TEAM_ID = os.environ.get('VERCEL_TEAM_ID', '')
PROJECT_ID = os.environ.get('VERCEL_PROJECT_ID', '')
PROJECT_NAME = 'temporary-agile-rowan-17dfxan'

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def deploy():
    print('[DEPLOY] Deploying to https://mabuy.vercel.app with all binary assets...')
    files_payload = []
    
    # 1. HTML
    html_path = os.path.join(ROOT_DIR, 'public', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    files_payload.append({'file': 'index.html', 'data': html_content})
    files_payload.append({'file': 'mainvest.html', 'data': html_content})

    # 2. Package & Vercel JSON
    package_json_path = os.path.join(ROOT_DIR, 'package.json')
    if os.path.exists(package_json_path):
        with open(package_json_path, 'r', encoding='utf-8') as f:
            files_payload.append({'file': 'package.json', 'data': f.read()})

    vercel_json_path = os.path.join(ROOT_DIR, 'vercel.json')
    if os.path.exists(vercel_json_path):
        with open(vercel_json_path, 'r', encoding='utf-8') as f:
            files_payload.append({'file': 'vercel.json', 'data': f.read()})

    # 3. Video Assets (easter_egg.mp4)
    video_path = os.path.join(ROOT_DIR, 'public', 'easter_egg.mp4')
    if os.path.exists(video_path):
        with open(video_path, 'rb') as f:
            b64_vid = base64.b64encode(f.read()).decode('utf-8')
        files_payload.append({'file': 'easter_egg.mp4', 'data': b64_vid, 'encoding': 'base64'})
        files_payload.append({'file': 'video/rdr2_easter_egg.mp4', 'data': b64_vid, 'encoding': 'base64'})
        print(f"Added easter_egg.mp4 ({os.path.getsize(video_path)} bytes) to deployment payload")

    # 3.2 Sunset image
    img_path = os.path.join(ROOT_DIR, 'public', 'images', 'rdr2_sunset.jpg')
    if os.path.exists(img_path):
        with open(img_path, 'rb') as f:
            b64_img = base64.b64encode(f.read()).decode('utf-8')
        files_payload.append({'file': 'images/rdr2_sunset.jpg', 'data': b64_img, 'encoding': 'base64'})
        print(f"Added rdr2_sunset.jpg ({os.path.getsize(img_path)} bytes) to deployment payload")

    # 4. API files
    api_dir = os.path.join(ROOT_DIR, 'api')
    if os.path.exists(api_dir):
        for fname in os.listdir(api_dir):
            fpath = os.path.join(api_dir, fname)
            if os.path.isfile(fpath) and fname.endswith('.js'):
                with open(fpath, 'r', encoding='utf-8') as f:
                    files_payload.append({'file': f'api/{fname}', 'data': f.read()})

    url = f'https://api.vercel.com/v13/deployments?teamId={TEAM_ID}'
    headers = {
        'Authorization': f'Bearer {VERCEL_TOKEN}',
        'Content-Type': 'application/json',
        'User-Agent': 'MaBuy-Deployer/5.0'
    }
    body = {
        'name': PROJECT_NAME,
        'project': PROJECT_ID,
        'target': 'production',
        'files': files_payload
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers=headers, method='POST')
    try:
        res = urllib.request.urlopen(req, timeout=60)
        data = json.loads(res.read().decode('utf-8'))
        print('[SUCCESS] DEPLOYED TO https://mabuy.vercel.app! State:', data.get('readyState'))
    except urllib.error.HTTPError as err:
        print("HTTP Error:", err.code, err.read().decode('utf-8'))

if __name__ == '__main__':
    deploy()
