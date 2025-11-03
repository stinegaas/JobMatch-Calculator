let punct = 0, gpa = 0, tech = 0, mood = 0, chem = 0, luck = 0;
let model = null;

async function loadModel() {
    const res = await fetch('model_export.json');
    model = await res.json();
}

// Activates button when clicked
function setActive(btn) {
    const group = btn.closest('div');
    if (!group) return;
    group.querySelectorAll('.my-button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
}

// Makes the buttons available for html
window.save_punct = (x, btn) => { punct = x; setActive(btn); };
window.save_gpa = (x, btn) => { gpa = x; setActive(btn); };
window.save_tech = (x, btn) => { tech = x; setActive(btn); };
window.save_mood = (x, btn) => { mood = x; setActive(btn); };
window.save_chem = (x, btn) => { chem = x; setActive(btn); };
window.save_luck = (x, btn) => { luck = x; setActive(btn); };

// Calculation
window.calculate = function () {
    const p = predict();
    if (p !== null && !isNaN(p)) {
        document.getElementById('prediction').textContent = //Updates prediction field
            (p * 100).toFixed(1) + '%';
    } else {
        document.getElementById('prediction').textContent = '–';
    }
};

// Estimates probability for calculation
function predict() {
    if (!model) return null;
    const x = [punct, gpa, tech, mood, chem, luck];

    // The data in the JSON-file is in model.scaler and model.logreg 
    const scale_ = model.scaler.scale_;
    const min_ = model.scaler.min_;
    const coef = model.logreg.coef;
    const intercept = model.logreg.intercept;

    // scales input
    const x_scaled = x.map((v, i) => v * scale_[i] + min_[i]);

    // linear combination to find z
    let z = intercept;
    for (let i = 0; i < coef.length; i++) {
        z += coef[i] * x_scaled[i];
    }

    const p = 1 / (1 + Math.exp(-z)); // logistic trans
    return p;
}

loadModel().then(predict);