document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("form");
    if (!form) return;

    const refreshBtn = document.getElementById("refreshBtn");
    const submitBtn = document.getElementById("submitBtn");
    const resultEl = document.getElementById("result");
    const textArea = document.getElementById("text");
    const countEl = document.getElementById("count");

    // Semua isi hasil (termasuk teks dari LLM) dirender via textContent, bukan innerHTML.
    const el = (tag, cls, text) => {
        const node = document.createElement(tag);
        if (cls) node.className = cls;
        if (text !== undefined) node.textContent = text;
        return node;
    };

    const showError = (msg) => {
        const box = el("div", "result-box error");
        box.appendChild(el("p", "result-meta", msg));
        resultEl.replaceChildren(box);
    };

    const WARNING = "Hasil ini adalah indikasi awal dari sistem otomatis dan bisa saja keliru. " +
        "Mohon tinjau kembali konteks teks sebelum mengambil kesimpulan atau tindakan.";
    const WARNING_DISAGREE = "Penilaian model ML dan AI berbeda, sehingga hasil ini perlu ditinjau lebih lanjut. ";

    const resultTitle = (data) =>
        data.result === "Tidak pasti" ? "Belum dapat dipastikan"
            : data.result === "KBGO" ? "Terindikasi KBGO" : "Tidak terindikasi KBGO";

    // Bila AI sepakat dengan hasil akhir, tidak ada angka persen; persen hanya tampil saat memakai model ML.
    const aiLeads = (data) => data.llm_used && data.llm_label === data.result;
    const mlPercent = (data) => Math.round(data.ml_probability * 100);

    // Peringatan selalu tampil di bawah setiap hasil, apa pun labelnya.
    const notice = (disagree) => {
        const n = el("div", "notice");
        n.appendChild(el("span", "notice-icon", "!"));
        n.appendChild(el("p", null, (disagree ? WARNING_DISAGREE : "") + WARNING));
        return n;
    };

    // Susun teks yang disalin: pesan pengguna + hasil analisis.
    const buildCopyText = (data, message) => {
        const lines = ["Pesan:", message.trim(), "", "Hasil: " + resultTitle(data)];
        if (data.result !== "Tidak pasti") {
            if (!aiLeads(data)) lines.push("Probabilitas KBGO menurut model: " + mlPercent(data) + "%");
            if (data.llm_used && data.category) lines.push("Kategori: " + data.category);
            if (data.llm_used && data.explanation) lines.push("Penjelasan: " + data.explanation);
        }
        return lines.join("\n");
    };

    const copyText = async (text) => {
        try {
            await navigator.clipboard.writeText(text);
        } catch (e) {
            // Fallback untuk halaman non-HTTPS (mis. diakses lewat IP jaringan lokal).
            const ta = document.createElement("textarea");
            ta.value = text;
            ta.style.position = "fixed";
            ta.style.opacity = "0";
            document.body.appendChild(ta);
            ta.select();
            const ok = document.execCommand("copy");
            ta.remove();
            if (!ok) throw e;
        }
    };

    const copyButton = (data, message) => {
        const btn = el("button", "copy-btn", "Salin");
        btn.type = "button";
        btn.addEventListener("click", async () => {
            try {
                await copyText(buildCopyText(data, message));
                btn.textContent = "Tersalin ✓";
            } catch (e) {
                btn.textContent = "Gagal menyalin";
            }
            setTimeout(() => { btn.textContent = "Salin"; }, 2000);
        });
        return btn;
    };

    const showResult = (data, message) => {
        const inconclusive = data.result === "Tidak pasti";
        const isKbgo = data.result === "KBGO";
        const box = el("div", "result-box " + (inconclusive ? "error" : isKbgo ? "kbgo" : "safe"));
        const head = el("div", "result-head");
        head.appendChild(el("p", "result-title", resultTitle(data)));
        head.appendChild(copyButton(data, message));
        box.appendChild(head);

        if (inconclusive) {
            box.appendChild(el("p", "result-meta",
                "Model tidak mengenali kata-kata dalam teks ini, sehingga tidak bisa memberi penilaian. " +
                "Coba tulis teks yang lebih lengkap."));
            box.appendChild(notice(false));
            resultEl.replaceChildren(box);
            return;
        }

        if (!aiLeads(data)) {
            const pct = mlPercent(data);
            box.appendChild(el("p", "result-meta", "Probabilitas KBGO menurut model: " + pct + "%"));
            const meter = el("div", "meter");
            const bar = el("span");
            bar.style.width = pct + "%";
            meter.appendChild(bar);
            box.appendChild(meter);
        }

        if (data.llm_used && data.category) {
            const chips = el("div", "chips");
            chips.appendChild(el("span", "chip cat", data.category));
            box.appendChild(chips);
        }

        if (data.llm_used && data.explanation) {
            const ex = el("div", "explain");
            ex.appendChild(el("b", null, "Penjelasan"));
            ex.appendChild(document.createTextNode(data.explanation));
            box.appendChild(ex);
        }
        box.appendChild(notice(data.needs_review));
        resultEl.replaceChildren(box);
    };

    textArea.addEventListener("input", () => { countEl.textContent = textArea.value.length; });

    refreshBtn.addEventListener("click", () => {
        textArea.value = "";
        countEl.textContent = "0";
        resultEl.replaceChildren();
        textArea.focus();
    });

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = textArea.value;
        if (!text.trim()) {
            showError("Silakan masukkan teks terlebih dahulu.");
            return;
        }

        submitBtn.textContent = "Memproses...";
        submitBtn.disabled = true;
        try {
            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            });
            const data = await res.json();
            if (!res.ok) showError(data.error || "Terjadi kesalahan.");
            else showResult(data, text);
        } catch (err) {
            showError("Gagal terhubung ke server.");
        } finally {
            submitBtn.textContent = "Analisis Sekarang";
            submitBtn.disabled = false;
        }
    });
});
