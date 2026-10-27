import csv
import io
import os
import secrets
import time
from collections import defaultdict, deque
from functools import wraps

import joblib
from dotenv import load_dotenv
from flask import (Flask, Response, abort, flash, g, jsonify, redirect,
                   render_template, request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv()  # baca credential dari .env sebelum modul lain membaca environment

import db  # noqa: E402
import llm  # noqa: E402

app = Flask(__name__)


def _load_secret_key():
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    os.makedirs(db.INSTANCE_DIR, exist_ok=True)
    path = os.path.join(db.INSTANCE_DIR, "secret_key")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip()


app.config.update(
    SECRET_KEY=_load_secret_key(),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE") == "1",
    MAX_CONTENT_LENGTH=64 * 1024,
    TEMPLATES_AUTO_RELOAD=os.environ.get("FLASK_RELOAD") == "1",
)
app.teardown_appcontext(db.close_db)
db.init_db()

model = joblib.load("model/logistic_regression_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

label_decoder = {0: "Non-KBGO", 1: "KBGO"}
INCONCLUSIVE = "Tidak pasti"
DELETE_ALL_PHRASE = "saya yakin untuk menghapus semua riwayat prediksi"
KBGO_CLASS_INDEX = list(model.classes_).index(1)

LOGIN_MAX_FAILS = 5
LOGIN_WINDOW_MIN = 10
PAGE_SIZE = 25
_hits = defaultdict(deque)


# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------
def log_activity(action, detail="", actor=None):
    if actor is None:
        actor = session.get("admin_username") or f"visitor:{g.get('visitor_id', '-')[:8]}"
    db.get_db().execute(
        "INSERT INTO activity_logs(ts, actor, ip, action, detail, user_agent) VALUES (?,?,?,?,?,?)",
        (db.now(), actor, request.remote_addr, action, detail[:500],
         (request.user_agent.string or "")[:200]),
    )
    db.get_db().commit()


def rate_limited(limit):
    now = time.time()
    q = _hits[request.remote_addr]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        return True
    q.append(now)
    return False


def to_int(value, default, lo, hi):
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def to_float(value, default, lo, hi):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("admin_id"):
            return redirect(url_for("admin_login"))
        return view(*args, **kwargs)
    return wrapped


@app.before_request
def before_request():
    g.settings = db.get_settings()
    g.new_visitor = False
    g.visitor_id = request.cookies.get("vid") or ""
    if len(g.visitor_id) != 32:
        g.visitor_id = secrets.token_hex(16)
        g.new_visitor = True

    if request.method == "POST" and request.path.startswith("/admin"):
        token = request.form.get("csrf", "")
        expected = session.get("csrf")
        if not expected or not secrets.compare_digest(token, expected):
            abort(400, "Token CSRF tidak valid")

    if g.settings["maintenance_mode"] == "1" and not (
        request.path.startswith(("/admin", "/static")) or session.get("admin_id")
    ):
        return render_template("maintenance.html"), 503


@app.after_request
def after_request(resp):
    if g.get("new_visitor"):
        resp.set_cookie("vid", g.visitor_id, max_age=365 * 24 * 3600,
                        httponly=True, samesite="Lax",
                        secure=app.config["SESSION_COOKIE_SECURE"])
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    return resp


@app.context_processor
def inject_globals():
    def page_url(n):
        args = request.args.to_dict()
        args["page"] = n
        return url_for(request.endpoint, **args)

    return {"settings": g.get("settings", db.DEFAULT_SETTINGS), "csrf_token": csrf_token,
            "admin_username": session.get("admin_username"), "page_url": page_url}


def page_view(name):
    if g.settings["log_page_views"] == "1":
        log_activity("page_view", name)


# --------------------------------------------------------------------------
# Halaman publik
# --------------------------------------------------------------------------
@app.route("/")
def index():
    page_view("/")
    return render_template("index.html")


@app.route("/about")
def about():
    page_view("/about")
    return render_template("about.html")


@app.route("/peneliti")
def peneliti():
    page_view("/peneliti")
    return render_template("peneliti.html")


@app.route("/predict", methods=["POST"])
def predict():
    s = g.settings
    if rate_limited(to_int(s["rate_limit_per_minute"], 20, 1, 1000)):
        return jsonify({"error": "Terlalu banyak permintaan, coba lagi sebentar lagi."}), 429

    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Teks tidak boleh kosong."}), 400
    text = text.strip()
    max_len = to_int(s["max_text_length"], 2000, 50, 20000)
    if len(text) > max_len:
        return jsonify({"error": f"Teks terlalu panjang (maksimal {max_len} karakter)."}), 400

    started = time.time()
    x = vectorizer.transform([text])
    ml_prob = float(model.predict_proba(x)[0][KBGO_CLASS_INDEX])
    ml_label = label_decoder[int(model.predict(x)[0])]
    # Kosakata TF-IDF terbatas: bila tidak ada satu pun kata yang dikenal, probabilitas hanya
    # berasal dari bias model (~50%) sehingga tidak boleh dianggap sebagai penilaian.
    ml_terms = int(x.nnz)
    ml_known = ml_terms > 0

    final_label = ml_label if ml_known else INCONCLUSIVE
    llm_result, llm_error, disagree = None, None, False
    if s["llm_enabled"] == "1":
        try:
            llm_result = llm.analyze(text, ml_label, ml_prob, s["llm_model"], ml_known)
            if not ml_known:
                final_label = llm_result["label"]  # ML tidak punya opini, LLM yang menentukan
            else:
                disagree = llm_result["label"] != ml_label
                if s["llm_mode"] == "hybrid" and disagree:
                    ml_conf = max(ml_prob, 1 - ml_prob)
                    if ml_conf < to_float(s["llm_override_threshold"], 0.75, 0.5, 1.0):
                        final_label = llm_result["label"]
        except llm.LLMError as exc:
            llm_error = str(exc)[:300]

    latency = int((time.time() - started) * 1000)
    d = db.get_db()
    d.execute(
        "INSERT INTO predictions(ts, visitor_id, ip, text, text_len, ml_label, ml_prob, llm_used,"
        " llm_label, llm_category, llm_confidence, llm_explanation, llm_error, final_label, disagree, latency_ms)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (db.now(), g.visitor_id, request.remote_addr,
         text if s["store_text"] == "1" else None, len(text), ml_label, ml_prob,
         int(llm_result is not None),
         llm_result and llm_result["label"], llm_result and llm_result["category"], llm_result and llm_result["confidence"],
         llm_result and llm_result["explanation"], llm_error, final_label, int(disagree), latency),
    )
    d.commit()
    log_activity("predict", f"{final_label} (ml={ml_label} {ml_prob:.2f} kata_dikenal={ml_terms}, llm={'ya' if llm_result else 'tidak'})")

    return jsonify({
        "result": final_label,
        "ml_label": ml_label,
        "ml_probability": round(ml_prob, 4),
        "ml_known": ml_known,
        "ml_terms": ml_terms,
        "llm_label": llm_result and llm_result["label"],
        "llm_confidence": llm_result and round(llm_result["confidence"], 2),
        "llm_used": llm_result is not None,
        "category": llm_result and llm_result["category"],
        "explanation": llm_result and llm_result["explanation"],
        "needs_review": disagree,
    })


# --------------------------------------------------------------------------
# Admin
# --------------------------------------------------------------------------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if session.get("admin_id"):
        return redirect(url_for("admin_dashboard"))
    if request.method == "POST":
        d = db.get_db()
        fails = d.execute(
            "SELECT COUNT(*) FROM activity_logs WHERE action='login_failed' AND ip=?"
            " AND ts > datetime('now', ?)",
            (request.remote_addr, f"-{LOGIN_WINDOW_MIN} minutes"),
        ).fetchone()[0]
        username = request.form.get("username", "").strip()
        if fails >= LOGIN_MAX_FAILS:
            flash(f"Terlalu banyak percobaan gagal. Coba lagi dalam {LOGIN_WINDOW_MIN} menit.", "error")
            return render_template("admin/login.html"), 429
        row = d.execute("SELECT * FROM admins WHERE username=?", (username,)).fetchone()
        if row and check_password_hash(row["password_hash"], request.form.get("password", "")):
            session.clear()
            session["admin_id"] = row["id"]
            session["admin_username"] = row["username"]
            d.execute("UPDATE admins SET last_login=? WHERE id=?", (db.now(), row["id"]))
            d.commit()
            log_activity("login", "Login admin berhasil")
            return redirect(url_for("admin_dashboard"))
        log_activity("login_failed", f"username={username[:50]}", actor=f"anon:{username[:50]}")
        flash("Username atau password salah.", "error")
    return render_template("admin/login.html")


@app.route("/admin/logout", methods=["POST"])
@admin_required
def admin_logout():
    log_activity("logout", "Logout admin")
    session.clear()
    return redirect(url_for("admin_login"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    d = db.get_db()
    one = lambda q, *a: d.execute(q, a).fetchone()[0]
    stats = {
        "total": one("SELECT COUNT(*) FROM predictions"),
        "kbgo": one("SELECT COUNT(*) FROM predictions WHERE final_label='KBGO'"),
        "today": one("SELECT COUNT(*) FROM predictions WHERE date(ts)=date('now')"),
        "visitors": one("SELECT COUNT(DISTINCT visitor_id) FROM predictions"),
        "llm": one("SELECT COUNT(*) FROM predictions WHERE llm_used=1"),
        "disagree": one("SELECT COUNT(*) FROM predictions WHERE disagree=1"),
        "llm_errors": one("SELECT COUNT(*) FROM predictions WHERE llm_error IS NOT NULL"),
        "avg_latency": one("SELECT COALESCE(AVG(latency_ms),0) FROM predictions"),
    }
    daily = d.execute(
        "SELECT date(ts) AS day, COUNT(*) AS total, SUM(final_label='KBGO') AS kbgo"
        " FROM predictions WHERE ts > datetime('now','-13 days') GROUP BY day ORDER BY day"
    ).fetchall()
    peak = max([r["total"] for r in daily] + [1])
    recent = d.execute("SELECT * FROM activity_logs ORDER BY id DESC LIMIT 8").fetchall()
    return render_template("admin/dashboard.html", stats=stats, daily=daily, peak=peak,
                           recent=recent, llm_ready=llm.is_configured())


def _paginate(base_sql, where, params, order_by="id DESC"):
    d = db.get_db()
    clause = f" WHERE {' AND '.join(where)}" if where else ""
    total = d.execute(f"SELECT COUNT(*) FROM ({base_sql}{clause})", params).fetchone()[0]
    page = to_int(request.args.get("page"), 1, 1, 10**6)
    rows = d.execute(f"{base_sql}{clause} ORDER BY {order_by} LIMIT ? OFFSET ?",
                     params + [PAGE_SIZE, (page - 1) * PAGE_SIZE]).fetchall()
    pages = max(1, -(-total // PAGE_SIZE))
    return rows, total, page, pages


@app.route("/admin/history")
@admin_required
def admin_history():
    where, params = [], []
    label = request.args.get("label", "")
    visitor = request.args.get("visitor", "").strip()
    q = request.args.get("q", "").strip()
    if label in ("KBGO", "Non-KBGO", INCONCLUSIVE):
        where.append("final_label=?"); params.append(label)
    if request.args.get("disagree") == "1":
        where.append("disagree=1")
    if visitor:
        where.append("visitor_id LIKE ?"); params.append(visitor + "%")
    if q:
        where.append("text LIKE ?"); params.append(f"%{q}%")
    rows, total, page, pages = _paginate("SELECT * FROM predictions", where, params)
    total_all = db.get_db().execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    return render_template("admin/history.html", rows=rows, total=total, page=page, pages=pages,
                           args=request.args, delete_phrase=DELETE_ALL_PHRASE, total_all=total_all)


@app.route("/admin/history/export")
@admin_required
def admin_history_export():
    rows = db.get_db().execute("SELECT * FROM predictions ORDER BY id").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(rows[0].keys() if rows else ["kosong"])
    for r in rows:
        # Awali sel berbahaya dengan ' agar tidak dieksekusi sebagai formula Excel.
        w.writerow([("'" + str(v)) if isinstance(v, str) and v[:1] in "=+-@\t\r" else v for v in tuple(r)])
    log_activity("export_history", f"{len(rows)} baris")
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=riwayat_prediksi.csv"})


@app.route("/admin/history/delete", methods=["POST"])
@admin_required
def admin_history_delete():
    pid = to_int(request.form.get("id"), 0, 0, 10**12)
    db.get_db().execute("DELETE FROM predictions WHERE id=?", (pid,))
    db.get_db().commit()
    log_activity("delete_prediction", f"id={pid}")
    flash("Riwayat dihapus.", "ok")
    return redirect(request.referrer or url_for("admin_history"))


@app.route("/admin/history/delete-all", methods=["POST"])
@admin_required
def admin_history_delete_all():
    typed = " ".join(request.form.get("confirm", "").lower().split())
    if typed != DELETE_ALL_PHRASE:
        flash("Kalimat konfirmasi tidak sesuai. Tidak ada riwayat yang dihapus.", "error")
        return redirect(url_for("admin_history"))
    d = db.get_db()
    count = d.execute("DELETE FROM predictions").rowcount
    d.commit()
    log_activity("delete_all_predictions", f"{count} riwayat prediksi dihapus")
    flash(f"{count} riwayat prediksi dihapus.", "ok")
    return redirect(url_for("admin_history"))


@app.route("/admin/logs")
@admin_required
def admin_logs():
    where, params = [], []
    action = request.args.get("action", "").strip()
    actor = request.args.get("actor", "").strip()
    if action:
        where.append("action=?"); params.append(action)
    if actor:
        where.append("actor LIKE ?"); params.append(f"%{actor}%")
    rows, total, page, pages = _paginate("SELECT * FROM activity_logs", where, params)
    actions = [r[0] for r in db.get_db().execute(
        "SELECT DISTINCT action FROM activity_logs ORDER BY action")]
    return render_template("admin/logs.html", rows=rows, total=total, page=page, pages=pages,
                           args=request.args, actions=actions)


@app.route("/admin/settings", methods=["GET", "POST"])
@admin_required
def admin_settings():
    if request.method == "POST":
        f = request.form
        new = {
            "site_name": f.get("site_name", "").strip()[:60] or db.DEFAULT_SETTINGS["site_name"],
            "announcement": f.get("announcement", "").strip()[:300],
            "maintenance_mode": "1" if f.get("maintenance_mode") else "0",
            "max_text_length": str(to_int(f.get("max_text_length"), 2000, 50, 20000)),
            "store_text": "1" if f.get("store_text") else "0",
            "log_page_views": "1" if f.get("log_page_views") else "0",
            "rate_limit_per_minute": str(to_int(f.get("rate_limit_per_minute"), 20, 1, 1000)),
            "llm_enabled": "1" if f.get("llm_enabled") else "0",
            "llm_mode": f.get("llm_mode") if f.get("llm_mode") in ("explain", "hybrid") else "hybrid",
            "llm_model": f.get("llm_model", "").strip()[:80] or db.DEFAULT_SETTINGS["llm_model"],
            "llm_override_threshold": str(to_float(f.get("llm_override_threshold"), 0.75, 0.5, 1.0)),
        }
        changed = [k for k, v in new.items() if g.settings.get(k) != v]
        for k, v in new.items():
            db.set_setting(k, v)
        db.get_db().commit()
        log_activity("settings_update", "Diubah: " + (", ".join(changed) or "tidak ada perubahan"))
        flash("Pengaturan disimpan.", "ok")
        return redirect(url_for("admin_settings"))
    return render_template("admin/settings.html", llm_ready=llm.is_configured())


@app.route("/admin/account", methods=["GET", "POST"])
@admin_required
def admin_account():
    d = db.get_db()
    if request.method == "POST":
        action = request.form.get("do")
        if action == "password":
            row = d.execute("SELECT * FROM admins WHERE id=?", (session["admin_id"],)).fetchone()
            new = request.form.get("new_password", "")
            if not check_password_hash(row["password_hash"], request.form.get("old_password", "")):
                flash("Password lama salah.", "error")
            elif len(new) < 10:
                flash("Password baru minimal 10 karakter.", "error")
            else:
                d.execute("UPDATE admins SET password_hash=? WHERE id=?",
                          (generate_password_hash(new), row["id"]))
                d.commit()
                log_activity("password_change", "Password diganti")
                flash("Password diganti.", "ok")
        elif action == "add":
            username = request.form.get("username", "").strip()
            pw = request.form.get("password", "")
            if not (3 <= len(username) <= 30) or len(pw) < 10:
                flash("Username 3-30 karakter dan password minimal 10 karakter.", "error")
            elif d.execute("SELECT 1 FROM admins WHERE username=?", (username,)).fetchone():
                flash("Username sudah dipakai.", "error")
            else:
                d.execute("INSERT INTO admins(username, password_hash, created_at) VALUES (?,?,?)",
                          (username, generate_password_hash(pw), db.now()))
                d.commit()
                log_activity("admin_add", f"username={username}")
                flash("Admin ditambahkan.", "ok")
        elif action == "delete":
            aid = to_int(request.form.get("id"), 0, 0, 10**9)
            if aid == session["admin_id"]:
                flash("Tidak bisa menghapus akun yang sedang dipakai.", "error")
            else:
                d.execute("DELETE FROM admins WHERE id=?", (aid,))
                d.commit()
                log_activity("admin_delete", f"id={aid}")
                flash("Admin dihapus.", "ok")
        return redirect(url_for("admin_account"))
    admins = d.execute("SELECT id, username, created_at, last_login FROM admins ORDER BY id").fetchall()
    return render_template("admin/account.html", admins=admins)


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1",
            use_reloader=os.environ.get("FLASK_RELOAD") == "1")
