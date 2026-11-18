#!/usr/bin/env python3
import os
import subprocess
import json

# Full updated HTML content
html_path = os.path.join(os.path.dirname(__file__), "..", "public", "index.html")
mainvest_path = os.path.join(os.path.dirname(__file__), "..", "public", "mainvest.html")

print("🚀 [1/3] Проверка файлов WebApp...")
if os.path.exists(mainvest_path):
    with open(mainvest_path, "r", encoding="utf-8") as src_f:
        content = src_f.read()
    with open(html_path, "w", encoding="utf-8") as dst_f:
        dst_f.write(content)
    print("✅ index.html успешно синхронизирован с новейшим mainvest.html")

public_dir = os.path.join(os.path.dirname(__file__), "..", "public")

print("📤 [2/3] Деплой на Vercel...")
try:
    proc = subprocess.run(
        ["npx", "vercel", "--prod", "--yes"],
        cwd=public_dir,
        capture_output=True,
        text=True,
        check=True
    )
    output = proc.stdout + proc.stderr
    print(output)
    
    # Extract Production URL
    prod_url = None
    for line in output.splitlines():
        if "https://" in line and ".vercel.app" in line:
            parts = line.strip().split()
            for p in parts:
                if p.startswith("https://") and ".vercel.app" in p:
                    prod_url = p.strip()
                    break
    
    if prod_url:
        print(f"🎯 Новый Production URL Vercel: {prod_url}")
        formatter_path = os.path.join(os.path.dirname(__file__), "..", "src", "presentation", "telegram_bot", "formatter.py")
        if os.path.exists(formatter_path):
            with open(formatter_path, "r", encoding="utf-8") as f:
                code = f.read()
            import re
            code = re.sub(r'MAINVEST_WEBAPP_URL",\s*"https://[^"]+"', f'MAINVEST_WEBAPP_URL", "{prod_url}"', code)
            with open(formatter_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"✅ Formatter обновлен с новым URL: {prod_url}")
except Exception as e:
    print("⚠️ Ошибка деплоя Vercel:", e)

print("🔄 [3/3] Перезапуск бота...")
os.system("systemctl restart mabuy-bot")
print("🎉 Готово! Проверь WebApp в Telegram!")
