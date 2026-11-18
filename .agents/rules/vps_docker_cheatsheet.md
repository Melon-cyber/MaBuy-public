# 🛠️ Шпаргалка по деплою обновлений на VPS через Docker & WinSCP

## 1. Загрузка обновлений (WinSCP):
Загрузите измененные файлы/папки из ПК на VPS в папку проекта (например, `/root/MaBuy`).

## 2. Пошаговые команды Docker на VPS (через SSH / PuTTY / PowerShell):

### Перезапуск и сборка с новыми правками:
```bash
docker-compose down && docker-compose up -d --build
```

### Проверка статуса контейнеров:
```bash
docker-compose ps
```

### Просмотр логов в реальном времени (tail):
```bash
docker-compose logs -f --tail=100
```

### Быстрая перезагрузка торгового демона без пересборки (если менялись только .py файлы):
```bash
docker-compose restart
```
