import os
os.environ['PANDAS_ARROW_DISABLED'] = '1'

import streamlit as st
import pandas as pd
import numpy as np

# Fonksiyonlar
def calculate_swara_weights(criteria, s_j_values):
    k_j = [1] + [1 + s for s in s_j_values[1:]]
    q_j = [1]
    for i in range(1, len(k_j)):
        q_j.append(q_j[i - 1] / k_j[i])
    weights = np.array(q_j) / sum(q_j)
    return dict(zip(criteria, weights))

def apply_topsis(df, weights_dict, criteria_types):
    matrix = df.to_numpy(dtype=float)
    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    weights = np.array([weights_dict[c] for c in df.columns])
    weighted_matrix = norm_matrix * weights
    ideal_pos, ideal_neg = [], []
    for j, crit in enumerate(df.columns):
        if criteria_types[crit] == "benefit":
            ideal_pos.append(weighted_matrix[:, j].max())
            ideal_neg.append(weighted_matrix[:, j].min())
        else:
            ideal_pos.append(weighted_matrix[:, j].min())
            ideal_neg.append(weighted_matrix[:, j].max())
    dist_pos = np.linalg.norm(weighted_matrix - ideal_pos, axis=1)
    dist_neg = np.linalg.norm(weighted_matrix - ideal_neg, axis=1)
    return dist_neg / (dist_pos + dist_neg)


# Başlangıç
st.title("Proje Önceliklendirmesi için Karar Destek Sistemi")

if "step" not in st.session_state:
    st.session_state.step = 1

if st.session_state.step == 1:
    uploaded_projects = st.file_uploader("1. Kriterlerinize göre puanlaması yapılmış projeler tablonuzu ekleyiniz. (İlk sütun talep kodlarını, ikinci sütun talep sahibi bölümleri ve diğer sütunlar kriterleri içermelidir. Dosya uzantısı .xlsx olmalıdır.)", type=["xlsx"])
    if uploaded_projects is not None:
        st.session_state.projects_df = pd.read_excel(uploaded_projects)
        st.session_state.criteria_names = st.session_state.projects_df.iloc[:, [2,3,4,6,8,10,12]].columns.tolist()
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.subheader("Kriterlerinizin türünü Fayda/Maliyet olarak belirtiniz. (Fayda=Artması tercih edilen olumlu etkiye sahip kriterler, Maliyet=Azalması tercih edilen olumsuz etkiye sahip kriterler)")
    st.session_state.criteria_types = {}
    for crit in st.session_state.criteria_names:
        choice = st.selectbox(f"{crit} kriteri", ["Fayda", "Maliyet"], key=f"type_{crit}")
        st.session_state.criteria_types[crit] = "benefit" if choice == "Fayda" else "cost"
    if st.button("Devam - Önem Sırası"):
        st.session_state.step = 3
        st.rerun()

elif st.session_state.step == 3:
    st.subheader("Her bir kriterin karar üzerindeki etkisini düşünerek, en önemli olandan en az önemli olana doğru bir sıralama yapınız. Bu sıralama, hangi kriterin öncelikli değerlendirileceğini belirlemek için kullanılacaktır.")

    if "selected_criteria" not in st.session_state:
        st.session_state.selected_criteria = []

    remaining_options = [c for c in st.session_state.criteria_names if c not in st.session_state.selected_criteria]

    selected = st.selectbox("Bir kriter seçin ve sıraya ekleyin:", options=["-- Seçiniz --"] + remaining_options, key="select_criteria")
    if selected != "-- Seçiniz --" and selected not in st.session_state.selected_criteria:
        st.session_state.selected_criteria.append(selected)
        st.rerun()

    if st.session_state.selected_criteria:
        st.markdown("### Seçilen Kriterler (Önem Sırası ile)")
        for i, c in enumerate(st.session_state.selected_criteria):
            st.write(f"{i+1}. {c}")

    if len(st.session_state.selected_criteria) == len(st.session_state.criteria_names):
        if st.button("Devam", key="swara_button_unique"):
            st.session_state.sorted_criteria = st.session_state.selected_criteria
            st.session_state.step = 4
            st.rerun()

elif st.session_state.step == 4:
    st.subheader("Her kriterin, kendinden bir üst sıradaki kritere göre ne kadar daha az önemli olduğunu sayı ile ifade ediniz. (Örneğin kriter kendinden bir üst sıradaki kritere göre %20 daha az önemliyse 0.2 girilmelidir. Kriterler eşit derecede önemli ise 0 yazabilirsiniz.")
    s_j_values = []
    for i, crit in enumerate(st.session_state.sorted_criteria):
        if i == 0:
            st.text(f"{crit} kriteri en önemli olarak belirlendiği için 0 kabul edilir.")
            s_j_values.append(0.0)
        else:
            val = st.number_input(f"{crit} kriteri için bir üst sıradaki kriter ile önem farkını girin", min_value=0.0, step=0.05, key=f"sj_{i}")
            s_j_values.append(val)
    if st.button("Ağırlıkları Hesapla"):
        st.session_state.s_j_values = s_j_values
        st.session_state.weights_dict = calculate_swara_weights(st.session_state.sorted_criteria, s_j_values)
        st.session_state.step = 5
        st.rerun()

elif st.session_state.step == 5:
    st.subheader("Ağırlıklar")
    for crit, w in st.session_state.weights_dict.items():
        st.write(f"{crit}: {w:.2%}")

    uploaded_limits = st.file_uploader("2. Talep sahibi bölümler için yıllık ayırmayı planladığınız maksimum bütçe dosaysını yükleyiniz. Dosya uzantısı .xlsx olmalıdır.", type=["xlsx"])
    total_budget_limit = st.number_input("Projeler için Ayrılan Toplam Yıllık Bütçe (₺)", min_value=0.0, step=1000.0)
    max_proj_per_dept = st.number_input("Talep sahibi bölümlerin bir yıl içinde gerçekleştirebileceği maksimsum proje sayısını giriniz.", min_value=1, step=1)
    if uploaded_limits is not None and st.button("Hesapla", key="run_calc"):
        limits_df = pd.read_excel(uploaded_limits)
        dept_limits = limits_df.set_index('Talep Sahibi Bölüm').to_dict(orient='index')

        df = st.session_state.projects_df.copy()
        df = df.set_index('Talep Kodu')
        score_data = df[st.session_state.sorted_criteria]

        scores = apply_topsis(score_data, st.session_state.weights_dict, st.session_state.criteria_types)

        df['Skor'] = scores
        sorted_df = df.sort_values(by='Skor', ascending=False)

        selected, dept_count, dept_cost = [], {}, {}
        for idx, row in sorted_df.iterrows():
            dept = row['Talep Sahibi Bölüm']
            cost = row['Tahmini Maliyet']
            limit = dept_limits.get(dept, {'Maksimum Bütçe': float('inf')})['Maksimum Bütçe']

            if dept_count.get(dept, 0) < max_proj_per_dept and dept_cost.get(dept, 0) + cost <= limit and sum(dept_cost.values()) + cost <= total_budget_limit:
                selected.append({
                    'Talep Kodu': idx,
                    'Talep Sahibi Bölüm': dept,
                    'Skor': row['Skor'],
                    'Tahmini Maliyet': cost
                })
                dept_count[dept] = dept_count.get(dept, 0) + 1
                dept_cost[dept] = dept_cost.get(dept, 0) + cost

        result_df = pd.DataFrame(selected)
        result_df['Skor'] = result_df['Skor'].map(lambda x: f"{x:.2f}")
        result_df = result_df[['Talep Sahibi Bölüm', 'Talep Kodu', 'Skor']].sort_values(by=['Talep Sahibi Bölüm', 'Skor'], ascending=[True, False])
        st.success(f"Seçilen {len(result_df)} proje")
        st.text(result_df.to_string(index=False))
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("CSV Olarak İndir", data=csv, file_name="sonuclar.csv", mime='text/csv')