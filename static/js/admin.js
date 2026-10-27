document.addEventListener("DOMContentLoaded", () => {
    // Dialog konfirmasi: tombol hapus baru aktif bila kalimat diketik persis.
    // Server tetap memverifikasi ulang kalimatnya.
    document.querySelectorAll("[data-open]").forEach((btn) => {
        const dialog = document.getElementById(btn.dataset.open);
        if (!dialog) return;
        btn.addEventListener("click", () => dialog.showModal());
        dialog.querySelectorAll("[data-close]").forEach((c) => c.addEventListener("click", () => dialog.close()));
        dialog.addEventListener("close", () => {
            const input = dialog.querySelector("input[name=confirm]");
            if (input) { input.value = ""; input.dispatchEvent(new Event("input")); }
        });
    });

    const input = document.getElementById("delete-confirm");
    const submit = document.getElementById("delete-submit");
    if (input && submit) {
        const norm = (v) => v.toLowerCase().split(/\s+/).filter(Boolean).join(" ");
        input.addEventListener("input", () => { submit.disabled = norm(input.value) !== input.dataset.phrase; });
    }
});
