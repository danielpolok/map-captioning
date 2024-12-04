import streamlit as st
import pandas as pd
from pathlib import Path
from datasets import load_from_disk
import re
import base64
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import json

# Load data (replace with the actual data path)
dataset_output_dir = Path("data/dataset/evaluated")
metrics_file_path = Path("data/eval/metrics.json")

dataset = load_from_disk(dataset_output_dir).to_pandas()

# Load metrics.json
with open(metrics_file_path, "r") as f:
    metrics = json.load(f)

# Extract Win Rates
win_rates = {k: v for k, v in metrics.items()}

# Initialize a set to store unique feature names
context_features = set()

# Iterate over each record in the dataset
for context in dataset["context"]:
    # Find all features using a regular expression
    features = re.findall(r"(\d+) ([a-zA-Z ]+)", context)
    # Add each feature with its amount to the set
    for amount, feature in features:
        amount = int(amount)  # Convert the amount to an integer
        # Remove the plural 's' if amount is greater than 1
        feature = feature.strip()
        if amount > 1 and feature.endswith("s"):
            feature = feature[:-1]
        # Add the processed feature to the set
        context_features.add(feature)

# Convert to a sorted list for easier reading
context_features = sorted(list(context_features))

# Streamlit app
st.title("Captioning Pipeline Evaluated Results")

# Top filters
st.header("Filter by Metadata Feature")

# Calculate the number of results for each feature
feature_counts = {
    feature: dataset["context"].apply(lambda x: feature in x).sum()
    for feature in context_features
}

st.markdown(
    """
    <style>
        .stMultiSelect [data-baseweb=select] span{
            max-width: 250px;
            font-size: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display options with counts in the top filter (full feature names)
context_type_filter = st.multiselect(
    "Context Type",
    options=[f"{feature} ({count})" for feature, count in feature_counts.items()],
    default=[],
    format_func=lambda x: x,  # Ensure full name is always displayed
    help="Hover over an option to see the full name",
)

# Display Win Rates
st.subheader("Win Rates")
for model, rate in win_rates.items():
    st.write(f"**{model}**: {rate}")

# Extract selected features (removing counts)
selected_features = [
    re.sub(r" \(\d+\)$", "", feature) for feature in context_type_filter
]

# Apply filters
if selected_features:
    filtered_df = dataset[
        dataset["context"].apply(
            lambda x: any(feature in x for feature in selected_features)
        )
    ]
else:
    filtered_df = dataset

# Display the number of filtered results
st.header("Results")
st.write(f"Number of results: {len(filtered_df)}")


# Function to decode base64 image and return a PIL image
def decode_base64_image(encoded_image):
    try:
        image_data = base64.b64decode(encoded_image)
        return Image.open(BytesIO(image_data))
    except (base64.binascii.Error, UnidentifiedImageError) as e:
        st.error(f"Error decoding image: {e}")
        return None


# Pagination
items_per_page = 10
num_pages = (len(filtered_df) + items_per_page - 1) // items_per_page
page_number = st.session_state.get("page_number", 1)

if "page_number" not in st.session_state:
    st.session_state["page_number"] = 1


def change_page(new_page):
    st.session_state["page_number"] = new_page


# Display table
start_idx = (page_number - 1) * items_per_page
end_idx = start_idx + items_per_page

column_names = dataset.columns
dataset_caption_columns = [
    column for column in column_names if re.match(r"caption_\w+", column)
]


def display_table():
    for idx, row in filtered_df.reset_index().iloc[start_idx:end_idx].iterrows():
        st.write(f"**Result #{start_idx + idx + 1}:**")
        col1, col2 = st.columns([3, 4])
        with col1:
            # Decode the base64 image and display it
            image = decode_base64_image(row["image_encoded"])
            if image:
                st.image(image, use_column_width=True)
        with col2:
            st.write(f"**Context:** {row['context']}")
            for column in dataset_caption_columns:
                with st.expander(f"**{column.capitalize()}**"):
                    st.write(row[column])
            st.write(f"**LLM Choice:** {row['llm_choice']}")
        st.markdown("---")


display_table()

# Pagination buttons
st.write("### Pages")
centered_cols = st.columns([1] + [0.5] * num_pages + [1], gap="small")
for i, col in enumerate(centered_cols[1:-1]):
    with col:
        if st.button(f"{i + 1}"):
            change_page(i + 1)
