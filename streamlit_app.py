
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==================================================
# 1. KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title="Dashboard Harga Pangan Kendari",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# 2. MEMUAT DATA
# ==================================================
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard_harga_pangan.csv")

    df["tanggal"] = pd.to_datetime(
        df["tanggal"], errors="coerce"
    )

    kolom_numerik = [
        "harga",
        "curah_hujan_mm",
        "suhu_rata_rata",
        "suhu_min",
        "suhu_max",
        "usd_idr"
    ]

    for kolom in kolom_numerik:
        if kolom in df.columns:
            df[kolom] = pd.to_numeric(
                df[kolom], errors="coerce"
            )

    return df.dropna(
        subset=["tanggal", "komoditas", "harga"]
    ).sort_values("tanggal")


@st.cache_resource
def load_model():
    model = joblib.load("model_rf.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor


try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal membaca dataset: {e}")
    st.stop()

# ==================================================
# 3. FILTER SIDEBAR
# ==================================================
st.sidebar.title("🔎 Filter Dashboard")
st.sidebar.caption(
    "Pilih komoditas dan periode untuk memfokuskan analisis."
)

komoditas_list = sorted(
    df["komoditas"].dropna().unique().tolist()
)

komoditas_pilihan = st.sidebar.multiselect(
    "🥬 Komoditas",
    options=komoditas_list,
    default=[],
    placeholder="Pilih satu atau beberapa komoditas"
)

tanggal_min = df["tanggal"].min().date()
tanggal_max = df["tanggal"].max().date()

rentang_tanggal = st.sidebar.date_input(
    "📅 Periode",
    value=(tanggal_min, tanggal_max),
    min_value=tanggal_min,
    max_value=tanggal_max,
    format="DD/MM/YYYY"
)

if isinstance(rentang_tanggal, (tuple, list)):
    if len(rentang_tanggal) == 2:
        tanggal_awal, tanggal_akhir = rentang_tanggal
    elif len(rentang_tanggal) == 1:
        tanggal_awal = tanggal_akhir = rentang_tanggal[0]
    else:
        tanggal_awal, tanggal_akhir = tanggal_min, tanggal_max
else:
    tanggal_awal = tanggal_akhir = rentang_tanggal

if tanggal_awal > tanggal_akhir:
    st.sidebar.error(
        "Tanggal awal tidak boleh melewati tanggal akhir."
    )
    st.stop()

# Filter berdasarkan periode
df_filter = df[
    (df["tanggal"].dt.date >= tanggal_awal)
    & (df["tanggal"].dt.date <= tanggal_akhir)
].copy()

# Jika komoditas dipilih, terapkan pilihan tersebut
if komoditas_pilihan:
    df_filter = df_filter[
        df_filter["komoditas"].isin(komoditas_pilihan)
    ].copy()

df_filter = df_filter.sort_values("tanggal")

# ==================================================
# 4. JUDUL DASHBOARD
# ==================================================
st.title("📊 Dashboard Harga Pangan Kota Kendari")

st.write(
    "Monitoring dan analisis harga pangan berdasarkan periode, "
    "komoditas, kondisi cuaca, dan kurs USD/IDR."
)

st.caption(
    f"Periode data tersedia: "
    f"{tanggal_min.strftime('%d %B %Y')} – "
    f"{tanggal_max.strftime('%d %B %Y')}"
)

# ==================================================
# 5. RINGKASAN DATA
# ==================================================
st.subheader("📌 Ringkasan Data")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Jumlah Data", f"{len(df_filter):,}")

with col2:
    st.metric(
        "Jumlah Komoditas",
        f"{df_filter['komoditas'].nunique():,}"
    )

with col3:
    rata_rata = df_filter["harga"].mean()
    st.metric(
        "Harga Rata-rata",
        f"Rp{rata_rata:,.0f}" if pd.notna(rata_rata) else "-"
    )

with col4:
    harga_tertinggi = df_filter["harga"].max()
    st.metric(
        "Harga Tertinggi",
        f"Rp{harga_tertinggi:,.0f}"
        if pd.notna(harga_tertinggi) else "-"
    )

st.divider()

# ==================================================
# 6. TAB DASHBOARD
# ==================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tren Harga",
    "🏆 Perbandingan",
    "🌦️ Cuaca dan Kurs",
    "🤖 Prediksi"
])

# ==================================================
# TAB 1: TREN HARGA
# ==================================================
with tab1:
    st.subheader("Pergerakan Harga Pangan")

    if df_filter.empty:
        st.warning("Tidak ada data pada periode yang dipilih.")

    else:
        if komoditas_pilihan:
            # Grafik mengikuti pilihan pengguna
            df_grafik = df_filter.copy()
            keterangan = "Grafik menampilkan komoditas yang dipilih."
        else:
            # Grafik default: 5 komoditas dengan rata-rata harga tertinggi
            komoditas_default = (
                df_filter.groupby("komoditas")["harga"]
                .mean()
                .nlargest(5)
                .index.tolist()
            )

            df_grafik = df_filter[
                df_filter["komoditas"].isin(komoditas_default)
            ].copy()

            keterangan = (
                "Grafik default menampilkan 5 komoditas "
                "dengan rata-rata harga tertinggi pada periode terpilih."
            )

        st.caption(keterangan)

        tren = (
            df_grafik.groupby(
                ["tanggal", "komoditas"], as_index=False
            )["harga"]
            .mean()
        )

        fig = px.line(
            tren,
            x="tanggal",
            y="harga",
            color="komoditas",
            markers=True,
            title="Tren Harga Pangan Berdasarkan Tanggal",
            labels={
                "tanggal": "Tanggal",
                "harga": "Harga (Rp)",
                "komoditas": "Komoditas"
            }
        )

        fig.update_layout(
            hovermode="x unified",
            legend_title_text="Komoditas"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detail Harga")

        st.dataframe(
            df_grafik[
                ["tanggal", "komoditas", "harga", "satuan"]
            ].sort_values(["tanggal", "komoditas"]),
            use_container_width=True,
            hide_index=True
        )

# ==================================================
# TAB 2: PERBANDINGAN HARGA
# ==================================================
with tab2:
    st.subheader("Perbandingan Harga Antar-Komoditas")

    if df_filter.empty:
        st.warning("Tidak ada data pada periode yang dipilih.")

    else:
        ringkasan = (
            df_filter.groupby("komoditas")["harga"]
            .agg(
                harga_rata_rata="mean",
                harga_minimum="min",
                harga_maksimum="max"
            )
            .reset_index()
            .sort_values("harga_rata_rata", ascending=True)
        )

        fig = px.bar(
            ringkasan,
            x="harga_rata_rata",
            y="komoditas",
            orientation="h",
            title="Rata-rata Harga Setiap Komoditas",
            labels={
                "harga_rata_rata": "Harga Rata-rata (Rp)",
                "komoditas": "Komoditas"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        tabel_ringkasan = ringkasan.rename(columns={
            "komoditas": "Komoditas",
            "harga_rata_rata": "Harga Rata-rata (Rp)",
            "harga_minimum": "Harga Minimum (Rp)",
            "harga_maksimum": "Harga Maksimum (Rp)"
        })

        st.subheader("Statistik Harga")

        st.dataframe(
            tabel_ringkasan.style.format({
                "Harga Rata-rata (Rp)": "{:,.0f}",
                "Harga Minimum (Rp)": "{:,.0f}",
                "Harga Maksimum (Rp)": "{:,.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )

# ==================================================
# TAB 3: CUACA DAN KURS
# ==================================================
with tab3:
    st.subheader("Kondisi Cuaca dan Kurs USD/IDR")

    kolom_cuaca = [
        "curah_hujan_mm",
        "suhu_rata_rata",
        "suhu_min",
        "suhu_max",
        "usd_idr"
    ]

    kolom_tersedia = [
        kolom for kolom in kolom_cuaca
        if kolom in df_filter.columns
    ]

    if df_filter.empty:
        st.warning("Tidak ada data pada periode yang dipilih.")

    elif not kolom_tersedia:
        st.warning("Kolom cuaca dan kurs tidak ditemukan.")

    else:
        # Ambil satu catatan per tanggal agar nilai cuaca
        # dan kurs tidak dihitung berulang untuk setiap komoditas.
        harian = (
            df_filter.groupby("tanggal")[kolom_tersedia]
            .first()
            .reset_index()
            .sort_values("tanggal")
        )

        col1, col2, col3 = st.columns(3)

        if "curah_hujan_mm" in harian.columns:
            nilai = harian["curah_hujan_mm"].iloc[-1]
            col1.metric(
                "Curah Hujan Terakhir",
                f"{nilai:.2f} mm" if pd.notna(nilai) else "-"
            )

        if "suhu_rata_rata" in harian.columns:
            nilai = harian["suhu_rata_rata"].iloc[-1]
            col2.metric(
                "Suhu Rata-rata Terakhir",
                f"{nilai:.1f} °C" if pd.notna(nilai) else "-"
            )

        if "usd_idr" in harian.columns:
            nilai = harian["usd_idr"].iloc[-1]
            col3.metric(
                "Kurs USD/IDR Terakhir",
                f"Rp{nilai:,.0f}" if pd.notna(nilai) else "-"
            )

        if "suhu_rata_rata" in harian.columns:
            st.write("🌡️ Perubahan Suhu Rata-rata")
            st.line_chart(
                harian.set_index("tanggal")[["suhu_rata_rata"]]
            )

        if "curah_hujan_mm" in harian.columns:
            st.write("🌧️ Perubahan Curah Hujan")
            st.bar_chart(
                harian.set_index("tanggal")[["curah_hujan_mm"]]
            )

        if "usd_idr" in harian.columns:
            st.write("💱 Perubahan Kurs USD/IDR")
            st.line_chart(
                harian.set_index("tanggal")[["usd_idr"]]
            )

# ==================================================
# TAB 4: PREDIKSI RANDOM FOREST
# ==================================================
with tab4:
    st.subheader("🤖 Prediksi Harga Pangan Hari Berikutnya")

    st.write(
        "Model Random Forest menggunakan fitur yang sama "
        "dengan proses pelatihan untuk menghasilkan estimasi harga."
    )

    try:
        model, preprocessor = load_model()

    except Exception as e:
        st.error(f"Model gagal dimuat: {e}")

    else:
        if df_filter.empty:
            st.warning("Tidak ada data untuk prediksi.")

        else:
            pilihan_prediksi = (
                sorted(df_filter["komoditas"].unique().tolist())
                if komoditas_pilihan
                else komoditas_list
            )

            komoditas_prediksi = st.selectbox(
                "Pilih komoditas untuk diprediksi",
                options=pilihan_prediksi
            )

            # Cari data terakhir dari komoditas yang dipilih
            # dalam periode filter yang aktif.
            data_komoditas = (
                df_filter[
                    df_filter["komoditas"] == komoditas_prediksi
                ]
                .sort_values("tanggal")
            )

            if data_komoditas.empty:
                st.warning("Data komoditas tidak ditemukan.")

            else:
                data_terakhir = data_komoditas.iloc[-1]

                col1, col2 = st.columns(2)

                col1.metric(
                    "Harga Terakhir",
                    f"Rp{data_terakhir['harga']:,.0f}"
                )

                col2.metric(
                    "Tanggal Data Terakhir",
                    data_terakhir["tanggal"].strftime("%d-%m-%Y")
                )

                if st.button(
                    "🔮 Prediksi Harga Berikutnya",
                    type="primary"
                ):
                    fitur_prediksi = pd.DataFrame([{
                        "komoditas": data_terakhir["komoditas"],
                        "harga": data_terakhir["harga"],
                        "curah_hujan_mm": data_terakhir["curah_hujan_mm"],
                        "suhu_rata_rata": data_terakhir["suhu_rata_rata"],
                        "suhu_min": data_terakhir["suhu_min"],
                        "suhu_max": data_terakhir["suhu_max"],
                        "usd_idr": data_terakhir["usd_idr"]
                    }])

                    try:
                        data_encoded = preprocessor.transform(
                            fitur_prediksi
                        )

                        hasil_prediksi = model.predict(
                            data_encoded
                        )[0]

                        st.success(
                            "Estimasi harga berikutnya: "
                            f"Rp{hasil_prediksi:,.2f}"
                        )

                        st.caption(
                            "Hasil merupakan estimasi model, bukan "
                            "harga aktual atau jaminan harga besok."
                        )

                    except Exception as e:
                        st.error(
                            f"Prediksi gagal dijalankan: {e}"
                        )

# ==================================================
# 7. FOOTER
# ==================================================
st.divider()

st.caption(
    "Proyek Data Science | Analisis Harga Pangan Kota Kendari"
)
