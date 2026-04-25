document.getElementById('checkBtn').addEventListener('click', async () => {
    const textInput = document.getElementById('inputText').value.trim();
    const resultCard = document.getElementById('resultCard');
    const resultText = document.getElementById('resultText');
    const sentimentIcon = document.getElementById('sentimentIcon');
    const btnText = document.querySelector('.btn-text');
    const loader = document.getElementById('loader');
    const analyzeBtn = document.getElementById('checkBtn');

    if (!textInput) {
        // Rung nhẹ textarea để báo lỗi
        document.getElementById('inputText').style.borderColor = '#f56565';
        setTimeout(() => document.getElementById('inputText').style.borderColor = '#e1e1e1', 1000);
        return;
    }

    // --- HIỂN THỊ TRẠNG THÁI LOADING ---
    btnText.style.opacity = '0';
    loader.style.display = 'block';
    analyzeBtn.disabled = true; // Khóa nút khi đang load
    resultCard.style.display = 'none'; // Giấu card kết quả cũ

    try {
        // Gửi data sang Backend app.py (Cổng 5000)
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: textInput })
        });

        if (!response.ok) throw new Error('Network error');
        const data = await response.json();

        // --- XỬ LÝ KẾT QUẢ ---
        // Gỡ bỏ tất cả các class màu cũ
        resultCard.classList.remove('positive', 'negative', 'neutral');

        // Dựa vào kết quả trả về từ Python
        if (data.result.includes("TÍCH CỰC")) {
            resultCard.classList.add('positive');
            sentimentIcon.innerText = "🟢"; // Icon mềm hơn
            resultText.innerText = "Tích Cực";
        } else if (data.result.includes("TIÊU CỰC")) {
            resultCard.classList.add('negative');
            sentimentIcon.innerText = "🔴";
            resultText.innerText = "Tiêu Cực";
        } else {
            resultCard.classList.add('neutral');
            sentimentIcon.innerText = "🟡";
            resultText.innerText = "Trung Tính";
        }

        // Hiện Card kết quả với hiệu ứng fadeIn
        resultCard.style.display = 'flex';

    } catch (error) {
        // Hiện lỗi kết nối
        resultCard.classList.remove('positive', 'neutral');
        resultCard.classList.add('negative');
        sentimentIcon.innerText = "❌";
        resultText.innerText = "Lỗi kết nối Backend (Cổng 5000)!";
        resultCard.style.display = 'flex';
    } finally {
        // --- TẮT LOADING ---
        btnText.style.opacity = '1';
        loader.style.display = 'none';
        analyzeBtn.disabled = false;
    }
});