/**
 * MaBuy Titan • Habit Tracker, Focus Pomodoro, Gamification & Multimedia Journal
 * Telegram Mini App (TMA) Client Engine
 */

(function () {
    'use strict';

    // 1. App Configuration & Telegram Init
    const tg = window.Telegram?.WebApp;
    if (tg) {
        tg.ready();
        tg.expand();
        try {
            tg.setHeaderColor('#0a0e17');
            tg.setBackgroundColor('#0a0e17');
        } catch (e) {}
    }

    // Determine current user ID (Default: owner's ID 100000001)
    const urlParams = new URLSearchParams(window.location.search);
    const userId = urlParams.get('user_id') || tg?.initDataUnsafe?.user?.id?.toString() || '100000001';
    const userName = tg?.initDataUnsafe?.user?.first_name || urlParams.get('name') || 'Владелец';

    // Global App State
    let appState = {
        userId: userId,
        userName: userName,
        todayData: null,
        gamification: null,
        journalEntries: [],
        activeTimerHabit: null,
        timerInterval: null,
        timerTotalSeconds: 45 * 60,
        timerRemainingSeconds: 45 * 60,
        timerIsRunning: false,
        uploadedMediaUrls: [],
        recordedAudioUrl: '',
        audioTracks: [],
        activePeriod: 'week',
        mediaRecorder: null,
        audioChunks: [],
        recordTimerInterval: null,
        recordSeconds: 0
    };

    // DOM Elements
    const elements = {
        userName: document.getElementById('userName'),
        rankBadge: document.getElementById('rankBadge'),
        rankIconHeader: document.getElementById('rankIconHeader'),
        xpBarFill: document.getElementById('xpBarFill'),
        xpText: document.getElementById('xpText'),
        nextRankText: document.getElementById('nextRankText'),
        streakCount: document.getElementById('streakCount'),
        todayDateLabel: document.getElementById('todayDateLabel'),
        completionRateBadge: document.getElementById('completionRateBadge'),
        completionSummary: document.getElementById('completionSummary'),
        dailyRingProgress: document.getElementById('dailyRingProgress'),
        ringText: document.getElementById('ringText'),
        habitsListContainer: document.getElementById('habitsListContainer'),
        btnOpenAddHabit: document.getElementById('btnOpenAddHabit'),
        btnResetHabits: document.getElementById('btnResetHabits'),
        
        // Modals
        habitModal: document.getElementById('habitModal'),
        btnCloseHabitModal: document.getElementById('btnCloseHabitModal'),
        habitForm: document.getElementById('habitForm'),
        habitIdInput: document.getElementById('habitIdInput'),
        habitTitleInput: document.getElementById('habitTitleInput'),
        habitIconInput: document.getElementById('habitIconInput'),
        habitCategoryInput: document.getElementById('habitCategoryInput'),
        habitTypeInput: document.getElementById('habitTypeInput'),
        habitValueInput: document.getElementById('habitValueInput'),
        habitDeadlineInput: document.getElementById('habitDeadlineInput'),

        // Timer
        focusTimerModal: document.getElementById('focusTimerModal'),
        btnCloseTimer: document.getElementById('btnCloseTimer'),
        timerHabitTitle: document.getElementById('timerHabitTitle'),
        timerCountdown: document.getElementById('timerCountdown'),
        timerStatusLabel: document.getElementById('timerStatusLabel'),
        timerCircleProgress: document.getElementById('timerCircleProgress'),
        btnStartPauseTimer: document.getElementById('btnStartPauseTimer'),
        btnResetTimer: document.getElementById('btnResetTimer'),
        timerMotivationQuote: document.getElementById('timerMotivationQuote'),

        // Motivation Banner
        motivationQuoteText: document.getElementById('motivationQuoteText'),
        motivationAuthor: document.getElementById('motivationAuthor'),
        motivationCategory: document.getElementById('motivationCategory'),
        btnNextQuote: document.getElementById('btnNextQuote'),

        // Journal
        journalContent: document.getElementById('journalContent'),
        btnRecordVoice: document.getElementById('btnRecordVoice'),
        recordBtnText: document.getElementById('recordBtnText'),
        recordTimer: document.getElementById('recordTimer'),
        audioPreview: document.getElementById('audioPreview'),
        todayAudioPlaylist: document.getElementById('todayAudioPlaylist'),
        todayAudioCount: document.getElementById('todayAudioCount'),
        todayAudioList: document.getElementById('todayAudioList'),
        mediaFileInput: document.getElementById('mediaFileInput'),
        btnSelectMedia: document.getElementById('btnSelectMedia'),
        mediaPreviewsContainer: document.getElementById('mediaPreviewsContainer'),
        btnSaveJournal: document.getElementById('btnSaveJournal'),
        journalTimeline: document.getElementById('journalTimeline'),
        entriesCountLabel: document.getElementById('entriesCountLabel'),

        // Smart AI Analytics
        analyticsGradeBadge: document.getElementById('analyticsGradeBadge'),
        statCompletionRate: document.getElementById('statCompletionRate'),
        statCompletedTasks: document.getElementById('statCompletedTasks'),
        statTotalHours: document.getElementById('statTotalHours'),
        statConsistency: document.getElementById('statConsistency'),
        statActiveDays: document.getElementById('statActiveDays'),
        statXpEarned: document.getElementById('statXpEarned'),
        aiCoachTitle: document.getElementById('aiCoachTitle'),
        aiCoachSummary: document.getElementById('aiCoachSummary'),
        aiCoachRec: document.getElementById('aiCoachRec'),
        categoryProgressContainer: document.getElementById('categoryProgressContainer'),

        // Heatmap & Ranks
        heatmapGrid: document.getElementById('heatmapGrid'),
        currentLvlBadge: document.getElementById('currentLvlBadge'),
        activeDaysStat: document.getElementById('activeDaysStat'),
        totalXpStat: document.getElementById('totalXpStat'),

        // Toast
        toastNotification: document.getElementById('toastNotification')
    };

    // Helper: Haptic Feedback
    function triggerHaptic(type = 'light') {
        if (tg?.HapticFeedback) {
            try {
                if (type === 'success' || type === 'warning' || type === 'error') {
                    tg.HapticFeedback.notificationOccurred(type);
                } else {
                    tg.HapticFeedback.impactOccurred(type);
                }
            } catch (e) {}
        }
    }

    // Helper: Web Audio Chime Synthesizer (No external mp3 needed)
    function playVictoryChime() {
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const now = ctx.currentTime;
            
            // Chord: C5, E5, G5, C6
            const freqs = [523.25, 659.25, 783.99, 1046.50];
            freqs.forEach((f, i) => {
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = 'sine';
                osc.frequency.setValueAtTime(f, now + (i * 0.08));
                
                gain.gain.setValueAtTime(0, now + (i * 0.08));
                gain.gain.linearRampToValueAtTime(0.3, now + (i * 0.08) + 0.04);
                gain.gain.exponentialRampToValueAtTime(0.001, now + (i * 0.08) + 1.2);
                
                osc.connect(gain);
                gain.connect(ctx.destination);
                
                osc.start(now + (i * 0.08));
                osc.stop(now + (i * 0.08) + 1.2);
            });
        } catch (e) {}
    }

    // Helper: Show Toast
    function showToast(msg) {
        if (!elements.toastNotification) return;
        elements.toastNotification.textContent = msg;
        elements.toastNotification.classList.remove('hidden');
        setTimeout(() => {
            elements.toastNotification.classList.add('hidden');
        }, 3000);
    }

    // =========================================================================
    // MOTIVATIONAL QUOTES & MINDSET ENGINE
    // =========================================================================

    const MOTIVATIONAL_QUOTES = [
        { text: "Боль дисциплины длится секунды. Боль сожаления длится всю жизнь.", author: "David Goggins", cat: "⚡️ ЖЕЛЕЗНАЯ ДИСЦИПЛИНА" },
        { text: "Работай так усердно, будто на твое место претендуют 100 лучших умов мира.", author: "Elon Musk", cat: "💼 МАЙНДСЕТ ФАУНДЕРА" },
        { text: "Великие победы не случаются импульсом, это сумма маленьких ежедневных спринтов.", author: "Vincent van Gogh", cat: "🎯 ЕЖЕДНЕВНЫЙ ГРАЙНД" },
        { text: "Твой главный соперник — вчерашний ты. Решай на 1 задачу больше, пиши на 10 строк кода чище.", author: "Kobe Bryant", cat: "🧮 SAT 1500+ & NU" },
        { text: "Глубокий фокус в течение 45 минут эффективнее 8 часов имитации бурной деятельности.", author: "Cal Newport", cat: "🧠 DEEP WORK" },
        { text: "Не жди вдохновения. Дисциплина сажает тебя за работу, пока остальные ищут оправдания.", author: "Marcus Aurelius", cat: "🗿 СТОИЦИЗМ & СИГМА" },
        { text: "Математика и алгоритмы не врут: побеждает тот, кто делает больше осознанных итераций.", author: "Satoshi Nakamoto", cat: "💻 FINANCIAL AI & ML" },
        { text: "Разница между тем, кто ты есть, и тем, кем хочешь стать — это то, что ты делаешь сегодня.", author: "Steve Jobs", cat: "💼 1% КЛУБ" },
        { text: "100 формул, 100 строк кода, 100 отжиманий каждый день. Секрет величия — в непрерывности.", author: "Saitama Mindset", cat: "🥋 ПУТЬ САЙТАМЫ" },
        { text: "Каждый решенный тест SAT приближает тебя к офферу мечты в Назарбаев Университет и топ-вузы.", author: "Academic Focus", cat: "🎓 NAZARBAYEV UNIVERSITY" },
        { text: "Если ты не построишь свою систему, кто-то наймет тебя обслуживать чужую.", author: "Tony Gaskins", cat: "💼 СТАТУС ФАУНДЕРА" },
        { text: "Твой мозг — это нейросеть. Обучай его на лучших датасетах: книгах, формулах и коде.", author: "Andrej Karpathy", cat: "💻 AI SYSTEMS" },
        { text: "Успех — это не случайность, это ежедневный выбор между ленью и величием.", author: "Alex Hormozi", cat: "⚡️ РОСТ & РЕЗУЛЬТАТ" },
        { text: "Дисциплина — это способность делать то, что нужно, даже когда совсем не хочется.", author: "Jocko Willink", cat: "🗿 ГИГАЧАД" }
    ];

    const TIMER_MOTIVATION_PUNCHLINES = [
        "Пока другие листают ленту, ты строишь свои 1500+ баллов и будущее.",
        "Чистейший Deep Work. 0 отвлечений, 100% концентрация.",
        "Каждая минута спринта превращается в опыт и силу мозга.",
        "Ты в шаге от решения сложнейшей задачи. Доведи до конца!",
        "Настоящие фаундеры куются в 45-минутных спринтах тишины."
    ];

    let currentQuoteIdx = 0;

    function renderMotivationQuote(idx = null) {
        if (!elements.motivationQuoteText) return;
        if (idx === null) {
            currentQuoteIdx = Math.floor(Math.random() * MOTIVATIONAL_QUOTES.length);
        } else {
            currentQuoteIdx = idx % MOTIVATIONAL_QUOTES.length;
        }
        const q = MOTIVATIONAL_QUOTES[currentQuoteIdx];
        
        elements.motivationQuoteText.style.opacity = '0';
        setTimeout(() => {
            elements.motivationQuoteText.textContent = `"${q.text}"`;
            if (elements.motivationAuthor) elements.motivationAuthor.textContent = `— ${q.author}`;
            if (elements.motivationCategory) elements.motivationCategory.textContent = q.cat;
            elements.motivationQuoteText.style.opacity = '1';
        }, 150);
    }

    function rotateTimerMotivation() {
        if (!elements.timerMotivationQuote) return;
        const randomPunch = TIMER_MOTIVATION_PUNCHLINES[Math.floor(Math.random() * TIMER_MOTIVATION_PUNCHLINES.length)];
        elements.timerMotivationQuote.style.opacity = '0';
        setTimeout(() => {
            elements.timerMotivationQuote.textContent = `"${randomPunch}"`;
            elements.timerMotivationQuote.style.opacity = '1';
        }, 200);
    }

    // =========================================================================
    // 2. DATA FETCHING & RENDERING
    // =========================================================================

    async function loadHabitsData() {
        try {
            const res = await fetch(`/api/v1/habits?user_id=${encodeURIComponent(appState.userId)}`);
            if (res.status === 403) {
                document.getElementById('vaultLockScreen')?.classList.remove('hidden');
                return;
            }
            if (res.ok) {
                const data = await res.json();
                appState.todayData = data;
                appState.gamification = data.gamification;
                renderHeader();
                renderHabitsList();
            }
        } catch (err) {
            console.error('Failed to load habits:', err);
        }
    }

    async function loadJournalData() {
        try {
            const res = await fetch(`/api/v1/journal?user_id=${encodeURIComponent(appState.userId)}&limit=30`);
            if (res.ok) {
                const data = await res.json();
                appState.journalEntries = data.entries || [];
                
                // Extract today's tracks into appState.audioTracks
                const todayStr = new Date().toISOString().split('T')[0];
                const todayEntry = appState.journalEntries.find(e => e.date_str === todayStr);
                if (todayEntry && todayEntry.audio_tracks) {
                    appState.audioTracks = todayEntry.audio_tracks;
                } else {
                    appState.audioTracks = [];
                }
                
                renderTodayAudioPlaylist();
                renderJournalTimeline();
            }
        } catch (err) {
            console.error('Failed to load journal:', err);
        }
    }

    async function loadAnalyticsData(period = null) {
        if (period) appState.activePeriod = period;
        const currentPeriod = appState.activePeriod || 'week';
        
        try {
            // 1. Fetch AI Analytics & Stats for current period
            const analyticsRes = await fetch(`/api/v1/habits/analytics?user_id=${encodeURIComponent(appState.userId)}&period=${currentPeriod}&lang=ru`);
            if (analyticsRes.ok) {
                const analyticsData = await analyticsRes.json();
                renderAnalytics(analyticsData);
            }

            // 2. Fetch Heatmap and Rank Gamification
            const statsRes = await fetch(`/api/v1/habits/stats?user_id=${encodeURIComponent(appState.userId)}&days=60`);
            if (statsRes.ok) {
                const data = await statsRes.json();
                renderHeatmap(data.activity_matrix);
                renderRankLadder(data.gamification);
            }
        } catch (err) {
            console.error('Failed to load analytics:', err);
        }
    }

    // =========================================================================
    // 3. UI RENDERING
    // =========================================================================

    function renderHeader() {
        if (!appState.gamification) return;
        const g = appState.gamification;
        const rank = g.rank || {};

        if (elements.userName) elements.userName.textContent = appState.userName;
        if (elements.rankBadge) elements.rankBadge.textContent = rank.title || 'NPC';
        if (elements.rankIconHeader) elements.rankIconHeader.textContent = rank.icon || '🥔';
        
        if (elements.xpText) elements.xpText.textContent = `${g.xp} / ${rank.next_rank_xp || 5000} XP`;
        if (elements.xpBarFill) elements.xpBarFill.style.width = `${rank.progress_pct || 0}%`;
        if (elements.nextRankText) elements.nextRankText.textContent = `➔ ${rank.next_title || 'Фаундер'}`;
        if (elements.streakCount) elements.streakCount.textContent = g.streak_current || 0;

        if (elements.totalXpStat) elements.totalXpStat.textContent = `⚡ ${g.xp} Всего XP`;

        // Daily Progress
        const today = appState.todayData;
        if (today) {
            const pct = today.completion_percentage || 0;
            if (elements.completionRateBadge) elements.completionRateBadge.textContent = `${pct}%`;
            if (elements.completionSummary) elements.completionSummary.textContent = `Выполнено ${today.completed_count} из ${today.total_habits} задач`;
            if (elements.dailyRingProgress) elements.dailyRingProgress.setAttribute('stroke-dasharray', `${pct}, 100`);
            if (elements.ringText) elements.ringText.textContent = `${Math.round(pct)}%`;
            
            // Format today's date in Russian
            const d = new Date();
            const options = { day: 'numeric', month: 'long', weekday: 'short' };
            if (elements.todayDateLabel) elements.todayDateLabel.textContent = d.toLocaleDateString('ru-RU', options);
        }
    }

    function renderHabitsList() {
        if (!elements.habitsListContainer || !appState.todayData) return;
        const habits = appState.todayData.habits || [];
        elements.habitsListContainer.innerHTML = '';

        if (habits.length === 0) {
            elements.habitsListContainer.innerHTML = `
                <div style="text-align: center; padding: 24px; color: var(--text-muted);">
                    <p>Нет активных привычек на сегодня.</p>
                    <button id="btnInitPreset" class="btn btn-primary" style="margin-top: 12px;">Загрузить академический пресет</button>
                </div>
            `;
            document.getElementById('btnInitPreset')?.addEventListener('click', resetDefaults);
            return;
        }

        habits.forEach(h => {
            const card = document.createElement('div');
            card.className = `habit-card ${h.is_completed ? 'completed' : ''}`;
            card.style.setProperty('--habit-color', h.color || '#38bdf8');

            const isMinutes = h.target_type === 'minutes';
            const deadlineHtml = h.deadline_time ? `<span class="habit-deadline">⏰ ${h.deadline_time}</span>` : '';
            const targetHtml = isMinutes ? `<span>⏱ ${h.logged_value || 0}/${h.target_value} ${h.unit}</span>` : `<span>🎯 ${h.unit}</span>`;

            card.innerHTML = `
                <div class="habit-left">
                    <div class="habit-icon-box" style="color: ${h.color};">${h.icon || '🎯'}</div>
                    <div class="habit-details">
                        <div class="habit-title">${escapeHtml(h.title)}</div>
                        <div class="habit-meta-row">
                            ${deadlineHtml}
                            ${targetHtml}
                        </div>
                    </div>
                </div>
                <div class="habit-right">
                    ${isMinutes && !h.is_completed ? `
                        <button class="btn-focus-sprint" data-habit-id="${h.habit_id}" data-habit-title="${escapeHtml(h.title)}">
                            ⚡ Спринт
                        </button>
                    ` : ''}
                    <div class="custom-checkbox ${h.is_completed ? 'checked' : ''}" data-habit-id="${h.habit_id}">
                        ${h.is_completed ? '✓' : ''}
                    </div>
                </div>
            `;

            // Toggle Checkbox Listener
            card.querySelector('.custom-checkbox')?.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleHabit(h.habit_id);
            });

            // Focus Sprint Button Listener
            card.querySelector('.btn-focus-sprint')?.addEventListener('click', (e) => {
                e.stopPropagation();
                openFocusTimer(h);
            });

            elements.habitsListContainer.appendChild(card);
        });
    }

    async function toggleHabit(habitId) {
        triggerHaptic('medium');
        try {
            const res = await fetch(`/api/v1/habits/${habitId}/log`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: appState.userId })
            });
            if (res.ok) {
                const data = await res.json();
                appState.todayData = data;
                appState.gamification = data.gamification;
                renderHeader();
                renderHabitsList();

                const updatedHabit = data.habits.find(x => x.habit_id === habitId);
                if (updatedHabit?.is_completed) {
                    triggerHaptic('success');
                    playVictoryChime();
                    showToast('+50 XP заработано! 🔥');
                }
            }
        } catch (err) {
            console.error('Toggle error:', err);
        }
    }

    async function resetDefaults() {
        if (!confirm('Сбросить привычки к академическому профилю (SAT Math, IELTS, ML, Спорт, Сон)?')) return;
        triggerHaptic('warning');
        try {
            const res = await fetch('/api/v1/habits/reset_defaults', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: appState.userId })
            });
            if (res.ok) {
                showToast('Академический профиль загружен! 🚀');
                await loadHabitsData();
            }
        } catch (err) {
            console.error('Reset error:', err);
        }
    }

    // =========================================================================
    // 4. FOCUS TIMER (POMODORO / DEEP WORK SPRINT)
    // =========================================================================

    function openFocusTimer(habit) {
        appState.activeTimerHabit = habit;
        if (elements.timerHabitTitle) {
            elements.timerHabitTitle.textContent = `⚡ Спринт: ${habit.title}`;
        }
        setTimerMode(45);
        elements.focusTimerModal?.classList.remove('hidden');
        triggerHaptic('light');
    }

    function setTimerMode(minutes) {
        clearInterval(appState.timerInterval);
        appState.timerIsRunning = false;
        appState.timerTotalSeconds = minutes * 60;
        appState.timerRemainingSeconds = minutes * 60;
        updateTimerDisplay();
        
        if (elements.btnStartPauseTimer) {
            elements.btnStartPauseTimer.textContent = '🚀 Начать Спринт';
            elements.btnStartPauseTimer.classList.remove('btn-secondary');
            elements.btnStartPauseTimer.classList.add('btn-primary');
        }
        if (elements.timerStatusLabel) elements.timerStatusLabel.textContent = 'ГОТОВ К СПРИНТУ';

        document.querySelectorAll('.timer-mode-btn').forEach(b => {
            b.classList.toggle('active', parseInt(b.dataset.minutes) === minutes);
        });
    }

    function updateTimerDisplay() {
        const mins = Math.floor(appState.timerRemainingSeconds / 60);
        const secs = appState.timerRemainingSeconds % 60;
        const str = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
        if (elements.timerCountdown) elements.timerCountdown.textContent = str;

        // SVG circle stroke-dashoffset: total perimeter = 2 * PI * 90 = ~565.48
        const total = 565.48;
        const progress = (appState.timerTotalSeconds - appState.timerRemainingSeconds) / appState.timerTotalSeconds;
        const offset = total * (1 - progress);
        if (elements.timerCircleProgress) {
            elements.timerCircleProgress.style.strokeDashoffset = offset;
        }
    }

    function toggleStartPauseTimer() {
        triggerHaptic('medium');
        if (appState.timerIsRunning) {
            // Pause
            clearInterval(appState.timerInterval);
            appState.timerIsRunning = false;
            elements.btnStartPauseTimer.textContent = '▶️ Продолжить';
            elements.timerStatusLabel.textContent = 'ПАУЗА';
        } else {
            // Start
            appState.timerIsRunning = true;
            elements.btnStartPauseTimer.textContent = '⏸ Пауза';
            elements.btnStartPauseTimer.classList.remove('btn-primary');
            elements.btnStartPauseTimer.classList.add('btn-secondary');
            elements.timerStatusLabel.textContent = 'ФОКУС-РЕЖИМ 🔥';

            appState.timerInterval = setInterval(() => {
                if (appState.timerRemainingSeconds > 0) {
                    appState.timerRemainingSeconds--;
                    updateTimerDisplay();
                    if (appState.timerRemainingSeconds % 45 === 0) {
                        rotateTimerMotivation();
                    }
                } else {
                    // Finished!
                    completeFocusTimer();
                }
            }, 1000);
        }
    }

    async function completeFocusTimer() {
        clearInterval(appState.timerInterval);
        appState.timerIsRunning = false;
        triggerHaptic('success');
        playVictoryChime();
        showToast('🔥 ПОБЕДА! Спринт закрыт! +75 XP! Твой мозг стал сильнее!');

        const habit = appState.activeTimerHabit;
        if (habit) {
            const minutesDone = Math.round(appState.timerTotalSeconds / 60);
            try {
                await fetch(`/api/v1/habits/${habit.habit_id}/log`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: appState.userId,
                        value: minutesDone,
                        is_completed: true
                    })
                });
                await loadHabitsData();
            } catch (e) {}
        }
        setTimeout(() => {
            elements.focusTimerModal?.classList.add('hidden');
        }, 2000);
    }

    // =========================================================================
    // 5. MULTIMEDIA JOURNAL & MEMORY ALBUM
    // =========================================================================

    // Voice Dictaphone with Multi-Track Appending
    async function toggleVoiceRecording() {
        triggerHaptic('medium');
        if (appState.mediaRecorder && appState.mediaRecorder.state === 'recording') {
            // Stop recording
            appState.mediaRecorder.stop();
            clearInterval(appState.recordTimerInterval);
            elements.btnRecordVoice?.classList.remove('recording');
            if (elements.recordBtnText) elements.recordBtnText.textContent = '🎙️ Дозаписать еще аудио';
            elements.recordTimer?.classList.add('hidden');
        } else {
            // Start recording
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                appState.audioChunks = [];
                appState.mediaRecorder = new MediaRecorder(stream);
                
                appState.mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) appState.audioChunks.push(e.data);
                };

                appState.mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(appState.audioChunks, { type: 'audio/webm' });
                    const recordDuration = appState.recordSeconds || 1;
                    // Upload & append audio to server
                    await uploadAudioFile(audioBlob, recordDuration);
                };

                appState.mediaRecorder.start();
                elements.btnRecordVoice?.classList.add('recording');
                if (elements.recordBtnText) elements.recordBtnText.textContent = '⏹ Остановить запись';
                elements.recordTimer?.classList.remove('hidden');
                
                appState.recordSeconds = 0;
                appState.recordTimerInterval = setInterval(() => {
                    appState.recordSeconds++;
                    const m = Math.floor(appState.recordSeconds / 60);
                    const s = appState.recordSeconds % 60;
                    if (elements.recordTimer) {
                        elements.recordTimer.textContent = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
                    }
                }, 1000);
            } catch (err) {
                console.warn('Microphone access unavailable or denied:', err);
                showToast('Микрофон недоступен');
            }
        }
    }

    async function uploadAudioFile(blob, duration = 0) {
        const todayStr = new Date().toISOString().split('T')[0];
        const formData = new FormData();
        formData.append('user_id', appState.userId);
        formData.append('file', blob, `voice_${Date.now()}.webm`);
        
        try {
            const res = await fetch('/api/v1/journal/upload', {
                method: 'POST',
                body: formData
            });
            if (res.ok) {
                const data = await res.json();
                const uploadedUrl = data.url;
                
                // Append directly to today's playlist
                const appendRes = await fetch('/api/v1/journal/audio', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: appState.userId,
                        date_str: todayStr,
                        audio_path: uploadedUrl,
                        duration: duration,
                        label: `Голосовая мысль #${appState.audioTracks.length + 1}`
                    })
                });

                if (appendRes.ok) {
                    const savedData = await appendRes.json();
                    if (savedData.entry && savedData.entry.audio_tracks) {
                        appState.audioTracks = savedData.entry.audio_tracks;
                    }
                    renderTodayAudioPlaylist();
                    triggerHaptic('success');
                    playVictoryChime();
                    showToast('Голосовая мысль добавлена в плейлист дня! 🎙️ (+50 XP)');
                    await loadJournalData();
                }
            }
        } catch (e) {
            console.error('Audio upload/append failed:', e);
            showToast('Ошибка сохранения аудио');
        }
    }

    async function deleteAudioTrack(trackId) {
        triggerHaptic('medium');
        const todayStr = new Date().toISOString().split('T')[0];
        try {
            const res = await fetch(`/api/v1/journal/audio/${encodeURIComponent(trackId)}?user_id=${encodeURIComponent(appState.userId)}&date=${todayStr}`, {
                method: 'DELETE'
            });
            if (res.ok) {
                appState.audioTracks = appState.audioTracks.filter(t => t.id !== trackId);
                renderTodayAudioPlaylist();
                showToast('Аудиозапись удалена');
                await loadJournalData();
            }
        } catch (e) {
            console.error('Delete audio track failed:', e);
        }
    }

    function renderTodayAudioPlaylist() {
        if (!elements.todayAudioPlaylist || !elements.todayAudioList) return;
        const tracks = appState.audioTracks || [];
        
        if (tracks.length === 0) {
            elements.todayAudioPlaylist.classList.add('hidden');
            return;
        }

        elements.todayAudioPlaylist.classList.remove('hidden');
        if (elements.todayAudioCount) elements.todayAudioCount.textContent = `${tracks.length} аудио`;
        elements.todayAudioList.innerHTML = '';

        tracks.forEach((track, idx) => {
            const item = document.createElement('div');
            item.className = 'audio-track-item';
            
            const durM = Math.floor((track.duration || 0) / 60);
            const durS = (track.duration || 0) % 60;
            const durStr = durM > 0 ? `${durM}м ${durS}с` : `${durS}с`;

            item.innerHTML = `
                <div class="track-info">
                    <span class="track-title">${escapeHtml(track.label || `Запись ${idx + 1}`)}</span>
                    <span class="track-dur">⏱ ${durStr}</span>
                </div>
                <audio controls src="${track.url}" class="track-player-mini"></audio>
                <button class="btn-delete-track" title="Удалить запись" data-track-id="${track.id}">🗑</button>
            `;

            item.querySelector('.btn-delete-track')?.addEventListener('click', () => {
                deleteAudioTrack(track.id);
            });

            elements.todayAudioList.appendChild(item);
        });
    }

    // Photo/Video Uploads
    async function handleMediaFiles(files) {
        if (!files || !files.length) return;
        for (const file of files) {
            const formData = new FormData();
            formData.append('user_id', appState.userId);
            formData.append('file', file);
            try {
                const res = await fetch('/api/v1/journal/upload', {
                    method: 'POST',
                    body: formData
                });
                if (res.ok) {
                    const data = await res.json();
                    appState.uploadedMediaUrls.push(data.url);
                    renderMediaPreviews();
                    triggerHaptic('light');
                }
            } catch (e) {
                console.error('Media upload error:', e);
            }
        }
    }

    function renderMediaPreviews() {
        if (!elements.mediaPreviewsContainer) return;
        elements.mediaPreviewsContainer.innerHTML = '';
        appState.uploadedMediaUrls.forEach(url => {
            const item = document.createElement('div');
            item.className = 'media-preview-item';
            item.innerHTML = `<img src="${url}" alt="media preview">`;
            elements.mediaPreviewsContainer.appendChild(item);
        });
    }

    async function saveJournalEntry() {
        const content = elements.journalContent?.value.trim();
        const activeMoodBtn = document.querySelector('.mood-btn.active');
        const mood = activeMoodBtn?.dataset.mood || '🔥';

        if (!content && appState.audioTracks.length === 0 && appState.uploadedMediaUrls.length === 0) {
            showToast('Напишите пару мыслей или запишите голос');
            return;
        }

        triggerHaptic('medium');
        const todayStr = new Date().toISOString().split('T')[0];

        try {
            const res = await fetch('/api/v1/journal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: appState.userId,
                    date_str: todayStr,
                    mood: mood,
                    content_text: content,
                    audio_tracks: appState.audioTracks,
                    media_paths: appState.uploadedMediaUrls
                })
            });

            if (res.ok) {
                triggerHaptic('success');
                playVictoryChime();
                showToast('Запись добавлена в Альбом Памяти! 📖 (+50 XP)');
                elements.journalContent.value = '';
                appState.uploadedMediaUrls = [];
                renderMediaPreviews();
                if (elements.audioPreview) elements.audioPreview.classList.add('hidden');
                
                await loadJournalData();
                await loadHabitsData();
            }
        } catch (err) {
            console.error('Failed to save journal:', err);
        }
    }

    function renderJournalTimeline() {
        if (!elements.journalTimeline) return;
        const entries = appState.journalEntries;
        if (elements.entriesCountLabel) elements.entriesCountLabel.textContent = `${entries.length} записей`;
        elements.journalTimeline.innerHTML = '';

        if (entries.length === 0) {
            elements.journalTimeline.innerHTML = `
                <div style="text-align: center; padding: 24px; color: var(--text-muted);">
                    <p>Альбом пока пуст. Запишите свой первый день выше!</p>
                </div>
            `;
            return;
        }

        entries.forEach(entry => {
            const card = document.createElement('div');
            card.className = 'memory-card';

            const mediaHtml = (entry.media_paths && entry.media_paths.length) ? `
                <div class="memory-media-gallery">
                    ${entry.media_paths.map(url => `<img src="${url}" alt="Memory photo" onclick="window.open('${url}', '_blank')">`).join('')}
                </div>
            ` : '';

            // Render all audio tracks for this entry
            let audioTracksHtml = '';
            const tracks = entry.audio_tracks || [];
            if (tracks.length > 0) {
                audioTracksHtml = `
                    <div class="memory-audio-playlist">
                        ${tracks.map(t => `
                            <div class="memory-audio-track">
                                <span class="track-badge">🎙️ ${escapeHtml(t.label || 'Голосовая заметка')}</span>
                                <audio controls src="${t.url}" class="memory-audio"></audio>
                            </div>
                        `).join('')}
                    </div>
                `;
            } else if (entry.audio_path) {
                audioTracksHtml = `<audio controls src="${entry.audio_path}" class="memory-audio"></audio>`;
            }

            card.innerHTML = `
                <div class="memory-header">
                    <span class="memory-date">🗓️ ${entry.date_str}</span>
                    <span class="badge badge-date">${entry.mood || '🔥'}</span>
                </div>
                ${entry.content_text ? `<div class="memory-text">${escapeHtml(entry.content_text)}</div>` : ''}
                ${audioTracksHtml}
                ${mediaHtml}
            `;
            elements.journalTimeline.appendChild(card);
        });
    }

    // =========================================================================
    // 6. SMART AI ANALYTICS, HEATMAP & RANKS
    // =========================================================================

    function renderAnalytics(data) {
        if (!data) return;
        const ai = data.ai_coach || {};
        const cats = data.category_breakdown || {};

        if (elements.analyticsGradeBadge) elements.analyticsGradeBadge.textContent = ai.grade || 'A+ ТИТАН';
        if (elements.statCompletionRate) elements.statCompletionRate.textContent = `${data.completion_rate || 0}%`;
        if (elements.statCompletedTasks) elements.statCompletedTasks.textContent = `${data.total_completed_tasks || 0}/${data.total_possible_tasks || 0} задач`;
        if (elements.statTotalHours) elements.statTotalHours.textContent = `${data.total_hours_logged || 0} ч`;
        if (elements.statConsistency) elements.statConsistency.textContent = `${data.consistency_score || 0}%`;
        if (elements.statActiveDays) elements.statActiveDays.textContent = `${data.total_active_days || 0} из ${data.days || 7} дн.`;
        if (elements.statXpEarned) elements.statXpEarned.textContent = `+${data.xp_earned || 0} XP`;

        if (elements.aiCoachTitle) elements.aiCoachTitle.textContent = ai.title || 'Тактический вердикт';
        if (elements.aiCoachSummary) elements.aiCoachSummary.innerHTML = ai.summary || '';
        if (elements.aiCoachRec) elements.aiCoachRec.textContent = ai.recommendation || '';

        // Category breakdown progress bars
        if (elements.categoryProgressContainer) {
            elements.categoryProgressContainer.innerHTML = '';
            Object.keys(cats).forEach(key => {
                const cat = cats[key];
                const hrs = (cat.minutes / 60.0).toFixed(1);
                const row = document.createElement('div');
                row.className = 'category-bar-row';
                
                const pctWidth = Math.min(100, Math.max(10, cat.completed * 15));
                row.innerHTML = `
                    <div class="cat-label-row">
                        <span>${cat.icon || '🎯'} ${cat.title || key}</span>
                        <span><b>${cat.completed} задач</b> (${hrs} ч)</span>
                    </div>
                    <div class="cat-progress-track">
                        <div class="cat-progress-fill" style="width: ${pctWidth}%;"></div>
                    </div>
                `;
                elements.categoryProgressContainer.appendChild(row);
            });
        }
    }

    function renderHeatmap(matrix) {
        if (!elements.heatmapGrid || !matrix) return;
        elements.heatmapGrid.innerHTML = '';
        let activeDays = 0;

        matrix.forEach(cell => {
            const sq = document.createElement('div');
            sq.className = `sq lvl-${cell.level}`;
            sq.title = `${cell.date}: выполнено ${cell.completed}/${cell.total} (${cell.pct}%)`;
            if (cell.completed > 0) activeDays++;
            elements.heatmapGrid.appendChild(sq);
        });

        if (elements.activeDaysStat) elements.activeDaysStat.textContent = `🔥 ${activeDays} активных дней`;
    }

    function renderRankLadder(gamification) {
        if (!gamification) return;
        const currentLvl = gamification.rank?.lvl || 1;
        if (elements.currentLvlBadge) elements.currentLvlBadge.textContent = `LVL ${currentLvl}`;

        document.querySelectorAll('.rank-step').forEach(step => {
            const lvl = parseInt(step.dataset.lvl);
            step.classList.toggle('unlocked', lvl < currentLvl);
            step.classList.toggle('current', lvl === currentLvl);
        });
    }

    // =========================================================================
    // 7. EVENT LISTENERS & NAVIGATION
    // =========================================================================

    // Navigation Tabs
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            triggerHaptic('light');
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const target = tab.dataset.tab;
            document.getElementById(target)?.classList.add('active');

            if (target === 'tabJournal') loadJournalData();
            if (target === 'tabAnalytics') loadAnalyticsData();
        });
    });

    // Period Toggle Buttons in AI Analytics Tab
    document.querySelectorAll('.period-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            triggerHaptic('light');
            document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const period = btn.dataset.period || 'week';
            loadAnalyticsData(period);
        });
    });

    // Add Habit Modal
    elements.btnOpenAddHabit?.addEventListener('click', () => {
        triggerHaptic('light');
        elements.habitIdInput.value = '';
        elements.habitTitleInput.value = '';
        elements.habitIconInput.value = '🎯';
        elements.habitModal?.classList.remove('hidden');
    });

    elements.btnCloseHabitModal?.addEventListener('click', () => {
        elements.habitModal?.classList.add('hidden');
    });

    elements.habitForm?.addEventListener('submit', async (e) => {
        e.preventDefault();
        triggerHaptic('medium');
        const habitData = {
            user_id: appState.userId,
            habit_id: elements.habitIdInput.value || undefined,
            title: elements.habitTitleInput.value.trim(),
            icon: elements.habitIconInput.value.trim() || '🎯',
            category: elements.habitCategoryInput.value,
            target_type: elements.habitTypeInput.value,
            target_value: parseInt(elements.habitValueInput.value) || 1,
            unit: elements.habitTypeInput.value === 'minutes' ? 'мин' : 'раз',
            deadline_time: elements.habitDeadlineInput.value
        };

        try {
            const res = await fetch('/api/v1/habits', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(habitData)
            });
            if (res.ok) {
                elements.habitModal?.classList.add('hidden');
                showToast('Привычка успешно сохранена!');
                await loadHabitsData();
            }
        } catch (err) {
            console.error('Save habit error:', err);
        }
    });

    // Reset Defaults
    elements.btnResetHabits?.addEventListener('click', resetDefaults);

    // Motivation Banner Refresh
    elements.btnNextQuote?.addEventListener('click', () => {
        triggerHaptic('light');
        renderMotivationQuote();
    });

    // Timer Controls
    elements.btnCloseTimer?.addEventListener('click', () => {
        clearInterval(appState.timerInterval);
        appState.timerIsRunning = false;
        elements.focusTimerModal?.classList.add('hidden');
    });

    document.querySelectorAll('.timer-mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            triggerHaptic('light');
            const mins = parseInt(btn.dataset.minutes);
            setTimerMode(mins);
        });
    });

    elements.btnStartPauseTimer?.addEventListener('click', toggleStartPauseTimer);
    elements.btnResetTimer?.addEventListener('click', () => {
        triggerHaptic('light');
        const activeBtn = document.querySelector('.timer-mode-btn.active');
        const mins = activeBtn ? parseInt(activeBtn.dataset.minutes) : 45;
        setTimerMode(mins);
    });

    // Journal Controls
    document.querySelectorAll('.mood-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            triggerHaptic('light');
            document.querySelectorAll('.mood-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    elements.btnRecordVoice?.addEventListener('click', toggleVoiceRecording);
    elements.btnSelectMedia?.addEventListener('click', () => elements.mediaFileInput?.click());
    elements.mediaFileInput?.addEventListener('change', (e) => handleMediaFiles(e.target.files));
    elements.btnSaveJournal?.addEventListener('click', saveJournalEntry);

    // Utility: Escape HTML
    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // Initial Load
    renderMotivationQuote();
    loadHabitsData();
})();
