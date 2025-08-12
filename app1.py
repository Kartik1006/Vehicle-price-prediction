import streamlit as st
import joblib
import numpy as np
from collections import defaultdict

st.set_page_config(page_title="Vehicle Price Prediction", layout="centered")

model = joblib.load('vehicle_price_model.pkl')

feature_names = None
n_expected = None
try:
    feature_names = getattr(model, "feature_name_", None)
    if callable(feature_names):
        feature_names = feature_names()
except Exception:
    feature_names = None

if not feature_names:
    try:
        feature_names = model.booster_.feature_name()
    except Exception:
        feature_names = None

try:
    if feature_names:
        n_expected = len(feature_names)
    else:
        n_expected = getattr(model, "n_features_", None)
        if n_expected is None:
            try:
                n_expected = model.booster_.num_feature()
            except Exception:
                n_expected = None
except Exception:
    n_expected = None

st.title('Vehicle Price Prediction')

st.markdown("Enter the required fields below. The model needs the same features used at training time — the app will supply defaults for unspecified features.")

year = st.number_input('Year', min_value=1980, max_value=2025, value=2018, step=1)
mileage = st.number_input('Mileage', min_value=0, max_value=1_000_000, value=45000, step=100)
cylinders = st.number_input('Cylinders', min_value=1, max_value=16, value=4, step=1)

if n_expected is None:
    st.warning("Model expected feature count could not be determined. Prediction may fail if the model truly requires a fixed-length input. Consider saving a pipeline with feature names when exporting the model.")
    # fallback: assume at least 3 features
    n_expected = max(3, getattr(model, "n_features_in_", 3))

X = np.zeros((1, n_expected), dtype=float)

def find_index_by_substr(name):
    if not feature_names:
        return None
    name = name.lower()
    for i, fn in enumerate(feature_names):
        if fn is None:
            continue
        fn_l = fn.lower()
        if fn_l == name:
            return i
    # second pass: substring match
    for i, fn in enumerate(feature_names):
        if fn is None:
            continue
        if name in fn.lower():
            return i
    return None

year_idx = find_index_by_substr('year')
mileage_idx = find_index_by_substr('mileage')
cylinders_idx = find_index_by_substr('cylinders')

if year_idx is not None:
    X[0, year_idx] = float(year)
if mileage_idx is not None:
    X[0, mileage_idx] = float(mileage)
if cylinders_idx is not None:
    X[0, cylinders_idx] = float(cylinders)

# Collect categorical one-hot groups (only when feature names are available)
groups = defaultdict(list)
if feature_names:
    for i, fn in enumerate(feature_names):
        if fn is None:
            continue
        fn = fn.strip()
        if '_' in fn:
            prefix = fn.split('_')[0]
            if prefix.lower() in ('feat', 'f', ''):
                continue
            groups[prefix].append((fn, i))

advanced = st.checkbox("Show advanced features (optional — improves accuracy if you know them)")
if advanced and groups:
    st.markdown("Set values for categorical features that were one-hot encoded at training time.")
    # pick top groups by size, exclude groups that are just year/mileage/cylinders
    candidate_groups = [g for g in groups.keys() if g.lower() not in ('year', 'mileage', 'cylinders')]
    candidate_groups = sorted(candidate_groups, key=lambda x: -len(groups[x]))[:6]
    for grp in candidate_groups:
        options = ['(leave as default)'] + [fn for fn, idx in groups[grp]]
        sel = st.selectbox(f"{grp} (choose one)", options, key=f"grp_{grp}")
        if sel != '(leave as default)':
            # find the index for selected one-hot and set to 1 (others remain 0)
            for fn, idx in groups[grp]:
                if fn == sel:
                    X[0, idx] = 1.0
                else:
                    X[0, idx] = 0.0

if (year_idx is None or mileage_idx is None or cylinders_idx is None) and n_expected >= 3:
    # find empty slots (zero valued) to attempt to place them
    empty_idxs = [i for i in range(n_expected) if X[0, i] == 0.0]
    used = 0
    if year_idx is None and used < len(empty_idxs):
        X[0, empty_idxs[used]] = float(year); used += 1
    if mileage_idx is None and used < len(empty_idxs):
        X[0, empty_idxs[used]] = float(mileage); used += 1
    if cylinders_idx is None and used < len(empty_idxs):
        X[0, empty_idxs[used]] = float(cylinders); used += 1

if st.button('Predict Price'):
    try:
        pred = model.predict(X)[0]
        st.success(f'Predicted Vehicle Price: ${pred:,.2f}')
        info_lines = []
        info_lines.append(f"Model expects {n_expected} features.")
        if feature_names:
            info_lines.append("Mapped inputs to feature names where found.")
        else:
            info_lines.append("No feature names found in the model; app used positional defaults.")
        st.info("  \n".join(info_lines))
    except Exception as e:
        st.error("Prediction failed. Error: " + str(e))
        st.write("Possible causes:")
        st.write("- Model truly needs the original preprocessing pipeline (one-hot, scaling).")
        st.write("- Feature order/names do not match what the model expects.")
        st.write("Recommended fixes:")
        st.write("1) At training time, save and export the full pipeline (preprocessing + model), e.g. using sklearn.pipeline and joblib. Then load that pipeline here and call pipeline.predict(X_dict_or_array).")
        st.write("2) If you can't re-run training, inspect model.feature_name_ and ensure your UI builds the exact feature vector (names and order) used during training.")
