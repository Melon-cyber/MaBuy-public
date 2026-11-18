// MaIA chat proxy (Vercel serverless function).
// - The Groq key lives only in the server environment (GROQ_API_KEY).
// - Same-origin by default. Other sites may call this endpoint only if they are listed in
//   ALLOWED_ORIGINS (comma-separated, e.g. "https://example.app"), so it is not an open proxy.
// - Input is size-limited.
// - When the model is unavailable it answers 503 with an error. It never invents an "analysis".
const LIMITS = { query: 2000, instructions: 1000, historyItems: 10, historyText: 500, alerts: 20 };

function allowedOrigin(origin) {
  const list = (process.env.ALLOWED_ORIGINS || '').split(',').map(s => s.trim()).filter(Boolean);
  return origin && list.includes(origin) ? origin : null;
}

export default async function handler(req, res) {
  const origin = allowedOrigin(req.headers && req.headers.origin);
  if (origin) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Vary', 'Origin');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  }

  if (req.method === 'OPTIONS') {
    return res.status(origin ? 204 : 403).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  let body = req.body;
  if (typeof body === 'string') {
    try { body = JSON.parse(body); } catch(e){}
  }
  body = (body && typeof body === 'object') ? body : {};

  const query = String(body.query || '').trim();
  const persona = String(body.persona || 'quant').toLowerCase();
  const depositNum = Number(body.deposit);
  const deposit = Number.isFinite(depositNum) && depositNum >= 0 ? Math.round(depositNum * 100) / 100 : 500;
  const alerts = (Array.isArray(body.alerts) ? body.alerts : ['BTC', 'ETH', 'SOL', 'XRP'])
    .map(a => String(a).toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 12))
    .filter(Boolean)
    .slice(0, LIMITS.alerts);
  const history = Array.isArray(body.history) ? body.history.slice(-LIMITS.historyItems) : [];
  const needsTitle = Boolean(body.needsTitle);
  const customInstructions = String(body.instructions || body.customInstructions || '').trim();

  if (!query) {
    return res.status(400).json({ error: 'Query is required' });
  }
  if (query.length > LIMITS.query || customInstructions.length > LIMITS.instructions) {
    return res.status(413).json({ error: 'Query or instructions too long' });
  }

  // Persona prompts
  const personaDescriptions = {
    tsundere: 'Ты — MaIA-Цундере (Tsundere). Дерзкая, острая на язык, саркастичная, но втайне безумно заботишься о депозите трейдера. Обзывай бакой, ругай за высокие плечи, но давай глубокий, точный профессиональный анализ рынка. Используй эмодзи (¬_¬) или ( >_< ).',
    yandere: 'Ты — MaIA-Яндере (Yandere). Одержимо влюблена в пользователя и его депозит! Считаешь его деньги СВОЕЙ собственностью, ревнуешь к другим ботам, говоришь сладко и пугающе-нежно с сердечками и нотками опасности: «Твой баланс принадлежит только мне, хи-хи ❤️🔪». Но анализ давай безупречно точный!',
    zen: 'Ты — MaIA-Дзен. Спокойный наставник долгосрочного накопления (DCA) и сложного процента. Успокаиваешь, отрезвляешь, убираешь суету и FOMO, объясняешь математическое ожидание и силу дисциплины. 🧘',
    quant: 'Ты — MaIA-Квант (PhD/CFA). Институциональный квантовый аналитик хедж-фонда. Считаешь матожидание, Шарп, Merton Jump Diffusion, L2 OFI и Quarter-Kelly сайзинг. Анализируй с холодным математическим превосходством. [●_●]',
    titan: 'Ты — MaIA-Titan 2077. Квантовый бортовой суперкомпьютер V5.1 из 2077 года. Кибер-терминальный стиль, матрицы вероятностей, сканирование рыночных аномалий.'
  };

  const personaDesc = personaDescriptions[persona] || personaDescriptions.quant;
  const groqApiKey = process.env.GROQ_API_KEY || '';
  if (!groqApiKey) {
    return res.status(503).json({ error: 'MaIA is not configured on this server (GROQ_API_KEY is missing)' });
  }

  const customInstBlock = customInstructions 
    ? `\n[🚨 ВАЖНЕЙШИЕ КАСТОМНЫЕ ПРАВИЛА И ИНСТРУКЦИИ ОТ ПОЛЬЗОВАТЕЛЯ (СТРОЖАЙШЕ И НЕУКОСНИТЕЛЬНО СЛЕДУЙ ИМ В КАЖДОМ ОТВЕТЕ!)]:\n${customInstructions}\n` 
    : '';

  const systemPrompt = `Ты — MaIA (Майя), официальный финансовый ИИ-компаньон MaBuy V5.1.
${personaDesc}
${customInstBlock}
[КОНТЕКСТ ПОЛЬЗОВАТЕЛЯ]:
• Текущий рабочий депозит: $${deposit}
• Активные монеты на радаре алертов: ${alerts.join(', ')}

[ИНСТРУКЦИЯ ПО АЛЕРТАМ]:
Если пользователь просит следить за монетой или поставить алерт (например, «следи за SOL» или «поставь алерт на BTC»), обязательно добавь в самый конец ответа тег [[ALERT_ADD:СИМВОЛ]], например [[ALERT_ADD:SOL]].
Если просит убрать алерт — добавь [[ALERT_REMOVE:СИМВОЛ]].

Отвечай емко, живо, экспертно и с соблюдением своего уникального характера. В самом конце ответа добавь дисклеймер: ⚖️ *Сгенерировано MaIA. Не ИИР.*`;

  // Clean recent history turns sent by the page (already capped to LIMITS.historyItems)
  const cleanTurns = history.map(m => ({
    role: m && m.role === 'user' ? 'user' : 'assistant',
    content: String((m && (m.text || m.content)) || '').replace(/\[\[ALERT_[^\]]+\]\]/g, '').slice(0, LIMITS.historyText)
  })).filter(m => m.content.length > 0);

  const messages = [
    { role: 'system', content: systemPrompt },
    ...cleanTurns,
    { role: 'user', content: query }
  ];

  let answer = '';
  let usedModel = '';
  let chatTitle = '';
  const startedAt = Date.now();
  const callGroq = async (payload, timeoutMs) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Authorization': `Bearer ${groqApiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    } finally {
      clearTimeout(timer);
    }
  };

  // Call Real Neural LLM via Groq
  try {
    const candidateModels = ['groq/compound-mini', 'groq/compound', 'openai/gpt-oss-120b', 'qwen/qwen3.6-27b'];
    
    for (const model of candidateModels) {
      if (Date.now() - startedAt > 8000) break; // stay within the serverless time limit
      try {
        const groqRes = await callGroq({ model, messages, temperature: 0.7, max_tokens: 600 }, 7000);
        if (groqRes.ok) {
          const data = await groqRes.json();
          let rawText = data?.choices?.[0]?.message?.content || '';
          rawText = rawText.replace(/<think>[\s\S]*?<\/think>/g, '').trim(); // drop reasoning tags
          if (rawText.length > 5) {
            answer = rawText;
            usedModel = model;
            break;
          }
        }
      } catch(err) {
        console.warn(`Model ${model} attempt failed:`, err && err.name);
      }
    }
  } catch(e) {
    console.error('Groq AI error:', e);
  }

  // No model answered: say so honestly instead of inventing an analysis.
  if (!answer) {
    return res.status(503).json({ error: 'MaIA is temporarily unavailable' });
  }

  // Automatic Chat Naming by MaIA
  if ((needsTitle || query.length > 5) && Date.now() - startedAt < 8000) {
    try {
      const titlePrompt = `Сгенерируй короткое (2-4 слова) емкое название диалога без кавычек и точки в конце на русском языке по первому вопросу пользователя: "${query.slice(0, 100)}". Напиши ТОЛЬКО само название (например: Анализ сигнала по BTC, Риск-менеджмент депозита, Настройка алертов SOL).`;
      const titleRes = await callGroq({
        model: 'groq/compound-mini',
        messages: [{ role: 'user', content: titlePrompt }],
        temperature: 0.3,
        max_tokens: 30
      }, 1500);

      if (titleRes.ok) {
        const titleData = await titleRes.json();
        let rawTitle = titleData?.choices?.[0]?.message?.content || '';
        rawTitle = rawTitle.replace(/<think>[\s\S]*?<\/think>/g, '').replace(/["'«»]/g, '').trim();
        if (rawTitle.length > 2 && rawTitle.length < 50) {
          chatTitle = rawTitle;
        }
      }
    } catch(titleErr) {
      console.warn('Auto-title generation notice:', titleErr);
    }
  }

  // If chatTitle not generated by AI, create smart title from query
  if (!chatTitle) {
    const qLower = query.toLowerCase();
    if (qLower.includes('btc') || qLower.includes('биткоин')) chatTitle = 'Анализ рынка BTC';
    else if (qLower.includes('eth') || qLower.includes('эфир')) chatTitle = 'Сетап по Ethereum';
    else if (qLower.includes('sol') || qLower.includes('солан')) chatTitle = 'Мониторинг Solana';
    else if (qLower.includes('алерт') || qLower.includes('следи')) chatTitle = 'Настройка радара алертов';
    else if (qLower.includes('риск') || qLower.includes('депозит')) chatTitle = 'Контроль рисков депозита';
    else {
      const words = query.split(/\s+/).slice(0, 3).join(' ');
      chatTitle = words.charAt(0).toUpperCase() + words.slice(1);
    }
  }

  return res.status(200).json({
    answer: answer,
    chatTitle: chatTitle,
    model: usedModel
  });
}
