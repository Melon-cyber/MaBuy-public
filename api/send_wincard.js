import https from 'https';

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method === 'GET') {
    return res.status(200).json({ status: 'ok', endpoint: 'send_wincard' });
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    let body = req.body;
    if (typeof body === 'string') {
      try { body = JSON.parse(body); } catch(e){}
    }
    body = body || {};

    const token = process.env.TELEGRAM_BOT_TOKEN;
    if (!token) {
      return res.status(500).json({ error: 'TELEGRAM_BOT_TOKEN is not configured on server' });
    }
    const userId = body.userId;
    if (!userId) {
      return res.status(400).json({ error: 'userId is required' });
    }
    const photoBase64 = body.photoBase64;
    const caption = body.caption || '📸 Твоя персональная Win-Card карточка сделки от MaBuy!';

    if (!photoBase64) {
      return res.status(400).json({ error: 'photoBase64 is required' });
    }

    const matches = photoBase64.match(/^data:([A-Za-z-+\/]+);base64,(.+)$/);
    let mimeType = 'image/png';
    let base64Clean = photoBase64;
    if (matches && matches.length === 3) {
      mimeType = matches[1];
      base64Clean = matches[2];
    } else {
      base64Clean = photoBase64.replace(/^data:image\/\w+;base64,/, '');
    }

    const imgBuffer = Buffer.from(base64Clean, 'base64');
    const boundary = '----MaBuyWinCardBoundary' + Date.now();
    const ext = mimeType.includes('jpeg') || mimeType.includes('jpg') ? 'jpg' : 'png';

    const part1 = Buffer.from(
      `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="chat_id"\r\n\r\n` +
      `${userId}\r\n` +
      `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="caption"\r\n\r\n` +
      `${caption}\r\n` +
      `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="photo"; filename="wincard.${ext}"\r\n` +
      `Content-Type: ${mimeType}\r\n\r\n`
    );

    const part2 = Buffer.from(`\r\n--${boundary}--\r\n`);
    const fullBody = Buffer.concat([part1, imgBuffer, part2]);

    const options = {
      hostname: 'api.telegram.org',
      port: 443,
      path: `/bot${token}/sendPhoto`,
      method: 'POST',
      headers: {
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
        'Content-Length': fullBody.length
      }
    };

    const tgResponse = await new Promise((resolve, reject) => {
      const tgReq = https.request(options, (tgRes) => {
        let raw = '';
        tgRes.on('data', chunk => { raw += chunk; });
        tgRes.on('end', () => {
          try {
            resolve(JSON.parse(raw));
          } catch(e) {
            resolve({ ok: false, error: 'JSON parse error', raw });
          }
        });
      });
      tgReq.on('error', reject);
      tgReq.write(fullBody);
      tgReq.end();
    });

    return res.status(200).json(tgResponse);
  } catch (err) {
    console.error('Error in send_wincard:', err);
    return res.status(500).json({ error: err.message });
  }
}
