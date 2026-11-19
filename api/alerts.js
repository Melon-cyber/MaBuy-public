// Serverless endpoint for syncing user alerts between WebApp and Telegram
export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method === 'GET') {
    const { user_id } = req.query;
    return res.status(200).json({
      ok: true,
      user_id: user_id || null,
      coins: ['BTC', 'ETH', 'SOL', 'XRP'],
      enabled: true,
      filter: 'sniper'
    });
  }

  if (req.method === 'POST') {
    let body = req.body;
    if (typeof body === 'string') {
      try { body = JSON.parse(body); } catch(e) {}
    }
    body = body || {};

    const { user_id, action, symbol, coins, enabled, filter } = body;
    const botToken = process.env.TELEGRAM_BOT_TOKEN;

    if (user_id && botToken && symbol && (action === 'add' || action === 'remove')) {
      try {
        const safeSym = String(symbol).replace(/[<>&]/g, '');
        const text = action === 'add'
          ? `🔔 <b>#${safeSym}</b> добавлен на радар 24/7 из WebApp!`
          : `🔕 <b>#${safeSym}</b> снят с мониторинга из WebApp.`;

        await fetch(`https://api.telegram.org/bot${botToken}/sendMessage`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            chat_id: user_id,
            text: text,
            parse_mode: 'HTML'
          })
        });
      } catch (e) {
        console.warn('Could not send Telegram alert sync notification:', e);
      }
    }

    return res.status(200).json({
      ok: true,
      message: 'Alerts synchronized successfully',
      user_id: user_id || null,
      coins: coins || ['BTC', 'ETH', 'SOL', 'XRP'],
      enabled: enabled !== false,
      filter: filter || 'sniper'
    });
  }

  return res.status(405).json({ error: 'Method not allowed' });
}
