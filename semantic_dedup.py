# import streamlit as st
# import pandas as pd
# import numpy as np
# import pyodbc
# import re
# import time
# from sentence_transformers import SentenceTransformer
# import faiss

# st.set_page_config(page_title="Semantic Deduplication", layout="wide")

# # --- Default Stopwords ---
# STOPWORDS = set([
#     "the", "and", "of", "in", "for", "on", "at", "to", "with", "by", "from", "or", "as", "is", "are", "was", "were",
#     "be", "this", "that", "&", "llc", "l.l.c", "a", "about", "above", "after", "again", "against", "all", "am", "an",
#     "any", "because", "been", "before", "being", "below", "between", "both", "but", "could", "did", "do", "does",
#     "doing", "down", "during", "each", "few", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
#     "herself", "him", "himself", "his", "how", "i", "if", "into", "it", "its", "itself", "me", "more", "most", "my",
#     "myself", "no", "nor", "not", "off", "once", "only", "other", "ought", "our", "ours", "ourselves", "out", "over",
#     "own", "same", "she", "should", "so", "some", "such", "than", "their", "theirs", "them", "themselves", "then",
#     "there", "these", "they", "those", "through", "too", "under", "until", "up", "very", "we", "what", "when", "where",
#     "which", "while", "who", "whom", "why", "would", "you", "your", "yours", "yourself", "yourselves",
#     "company", "incorporated", "inc", "ltd", "limited", "sole proprietorship", "w.l.l", "liability", "LLC", "LLP"
# ])

# # --- Preprocess Function ---
# def preprocess_text(text, stopwords):
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', ' ', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     for phrase in sorted(stopwords, key=lambda x: -len(x)):
#         pattern = r'\b' + re.escape(phrase) + r'\b'
#         text = re.sub(pattern, '', text)
#     return re.sub(r'\s+', ' ', text).strip()

# def get_connection(server, db, user, pwd):
#     conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={db};UID={user};PWD={pwd};"
#     return pyodbc.connect(conn_str)

# def group_by_similarity(vectors, threshold):
#     dim = vectors.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     index.add(vectors)
#     sim, idx = index.search(vectors, 10)

#     n = len(vectors)
#     assigned = [-1] * n
#     scores = [0.0] * n
#     gid = 1

#     for i in range(n):
#         if assigned[i] != -1:
#             continue
#         assigned[i] = gid
#         max_sim = 0
#         for j, s in zip(idx[i], sim[i]):
#             if i != j and s >= threshold and assigned[j] == -1:
#                 assigned[j] = gid
#                 max_sim = max(max_sim, s)
#         scores[i] = round(max_sim * 100, 2)
#         gid += 1

#     return assigned, scores

# def write_to_sql(df, table_name, cursor, cnxn, group_col, score_col, unique_col):
#     cursor.execute(f"SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?", table_name)
#     if not cursor.fetchone():
#         col_defs = ', '.join([
#             f"[{col}] VARCHAR(MAX)"
#             for col in df.columns if col not in [group_col, score_col, unique_col]
#         ])
#         create_stmt = (
#             f"CREATE TABLE {table_name} ({col_defs}, "
#             f"{group_col} INT, {score_col} FLOAT, {unique_col} BIT)"
#         )
#         cursor.execute(create_stmt)
#         cnxn.commit()

#     cursor.execute(f"DELETE FROM {table_name}")
#     cnxn.commit()

#     insert_cols = df.columns.tolist()
#     insert_query = f"INSERT INTO {table_name} ({', '.join(insert_cols)}) VALUES ({', '.join(['?'] * len(insert_cols))})"

#     records = [
#         [str(row[col]) if col not in [group_col, score_col, unique_col]
#          else float(row[col]) if col == score_col
#          else int(row[col]) for col in insert_cols]
#         for _, row in df.iterrows()
#     ]

#     fast_cursor = cnxn.cursor()
#     fast_cursor.fast_executemany = True
#     for i in range(0, len(records), 1000):
#         fast_cursor.executemany(insert_query, records[i:i + 1000])
#         cnxn.commit()

# # --- UI ---
# st.title("🔎 AskSyniti - Semantic Deduplication")

# with st.form("dedup_form"):
#     st.header("🔐 SQL Server Connection")
#     server = st.text_input("Server")
#     db = st.text_input("Database")
#     user = st.text_input("Username")
#     pwd = st.text_input("Password", type="password")

#     st.header("🧾 Table & Parameters")
#     input_table = st.text_input("Input Table Name")
#     output_table = st.text_input("Output Table Name")
#     key_column = st.text_input("Key Column(s) (comma-separated)")
#     groupby_column = st.text_input("GroupBy Column(s) (optional, comma-separated)", "")
#     custom_stopwords = st.text_area("Additional Stopwords (comma-separated)", "")
#     threshold = st.slider("Match Score Threshold", 0.7, 1.0, 0.85, 0.01)

#     submitted = st.form_submit_button("Run")

# if submitted:
#     try:
#         st.info("🔌 Connecting to SQL Server...")
#         start = time.time()
#         cnxn = get_connection(server, db, user, pwd)
#         cursor = cnxn.cursor()
#         st.success("✅ Connected!")

#         df = pd.read_sql(f"SELECT * FROM {input_table}", cnxn)

#         keys = [k.strip() for k in key_column.split(",")]
#         df["_key"] = df[keys].astype(str).agg(" ".join, axis=1).fillna("")

#         # Merge default stopwords with custom ones
#         custom_stop_set = set([word.strip().lower() for word in custom_stopwords.split(",") if word.strip()])
#         all_stopwords = STOPWORDS.union(custom_stop_set)

#         df["_processed"] = df["_key"].apply(lambda x: preprocess_text(x, all_stopwords))

#         st.info("📦 Loading embedding model...")
#         model = SentenceTransformer('all-MiniLM-L6-v2')

#         group_col = "GroupID"
#         score_col = "MatchScore"
#         unique_col = "IsUnique"
#         all_groups = []
#         gid_counter = 1

#         groupby_columns = [col.strip() for col in groupby_column.split(",") if col.strip()]
#         if groupby_columns:
#             st.info("🧠 Grouping records by selected column(s)...")
#             for _, group_df in df.groupby(groupby_columns):
#                 if len(group_df) == 0:
#                     continue
#                 emb = model.encode(group_df["_processed"].tolist(), convert_to_numpy=True, normalize_embeddings=True)
#                 local_gids, scores = group_by_similarity(emb, threshold)
#                 gid_map = {}
#                 adjusted_ids = []
#                 for gid in local_gids:
#                     if gid not in gid_map:
#                         gid_map[gid] = gid_counter
#                         gid_counter += 1
#                     adjusted_ids.append(gid_map[gid])
#                 group_df[group_col] = adjusted_ids
#                 group_df[score_col] = scores
#                 all_groups.append(group_df)
#         else:
#             st.info("🧠 Processing all records together...")
#             emb = model.encode(df["_processed"].tolist(), convert_to_numpy=True, normalize_embeddings=True)
#             gids, scores = group_by_similarity(emb, threshold)
#             df[group_col] = gids
#             df[score_col] = scores
#             all_groups.append(df)

#         final_df = pd.concat(all_groups, ignore_index=True)
#         final_df[unique_col] = final_df.groupby(group_col)[group_col].transform('count') == 1
#         final_df[unique_col] = final_df[unique_col].astype(int)

#         st.success("✅ Deduplication done")
#         st.dataframe(final_df.head(30))

#         st.info("📤 Writing output to SQL...")
#         write_to_sql(final_df.drop(columns=["_key", "_processed"]), output_table, cursor, cnxn, group_col, score_col, unique_col)
#         st.success(f"🏁 Done! Time taken: {time.time() - start:.2f}s")

#         cursor.close()
#         cnxn.close()

#     except Exception as e:
#         st.error(f"❌ Error: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import pyodbc
import re
import time
from sentence_transformers import SentenceTransformer
import faiss
from concurrent.futures import ThreadPoolExecutor
from itertools import count

st.set_page_config(page_title="Semantic Deduplication", layout="wide")

# --- Stopwords ---
DEFAULT_STOPWORDS = set([
    "the", "and", "of", "in", "for", "on", "at", "to", "with", "by", "from", "or", "as", "is", "are", "was", "were",
    "be", "this", "that", "&", "llc", "l.l.c", "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "any", "because", "been", "before", "being", "below", "between", "both", "but", "could", "did", "do", "does",
    "doing", "down", "during", "each", "few", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "into", "it", "its", "itself", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "off", "once", "only", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "those", "through", "too", "under", "until", "up", "very", "we", "what", "when", "where",
    "which", "while", "who", "whom", "why", "would", "you", "your", "yours", "yourself", "yourselves",
    "company", "incorporated", "inc", "ltd", "limited", "sole proprietorship", "w.l.l", "liability", "LLC", "LLP"
])

def preprocess_text(text, stopwords):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    for phrase in sorted(stopwords, key=lambda x: -len(x)):
        pattern = r'\b' + re.escape(phrase) + r'\b'
        text = re.sub(pattern, '', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_connection(server, db, user, pwd):
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={db};UID={user};PWD={pwd};"
    return pyodbc.connect(conn_str)

def group_by_similarity(vectors, threshold):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    sim, idx = index.search(vectors, 10)

    n = len(vectors)
    assigned = [-1] * n
    scores = [0.0] * n
    gid = 1

    for i in range(n):
        if assigned[i] != -1:
            continue
        assigned[i] = gid
        max_sim = 0
        for j, s in zip(idx[i], sim[i]):
            if i != j and s >= threshold and assigned[j] == -1:
                assigned[j] = gid
                max_sim = max(max_sim, s)
        scores[i] = round(max_sim * 100, 2)
        gid += 1

    return assigned, scores

def write_to_sql(df, table_name, cursor, cnxn, group_col, score_col, unique_col):
    cursor.execute(f"SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?", table_name)
    if not cursor.fetchone():
        col_defs = ', '.join([f"[{col}] VARCHAR(MAX)" for col in df.columns if col not in [group_col, score_col, unique_col]])
        create_stmt = f"CREATE TABLE {table_name} ({col_defs}, {group_col} INT, {score_col} FLOAT, {unique_col} BIT)"
        cursor.execute(create_stmt)
        cnxn.commit()

    cursor.execute(f"DELETE FROM {table_name}")
    cnxn.commit()

    insert_cols = df.columns.tolist()
    insert_query = f"INSERT INTO {table_name} ({', '.join(insert_cols)}) VALUES ({', '.join(['?'] * len(insert_cols))})"

    records = [
        [str(row[col]) if col not in [group_col, score_col, unique_col]
         else float(row[col]) if col == score_col
         else int(row[col]) for col in insert_cols]
        for _, row in df.iterrows()
    ]

    fast_cursor = cnxn.cursor()
    fast_cursor.fast_executemany = True
    for i in range(0, len(records), 1000):
        fast_cursor.executemany(insert_query, records[i:i + 1000])
        cnxn.commit()

# Load model only once
model = SentenceTransformer('all-MiniLM-L6-v2')

# UI
st.title("🔎 AskSyniti - Semantic Deduplication")

with st.form("dedup_form"):
    st.header("🔐 SQL Server Connection")
    server = st.text_input("Server")
    db = st.text_input("Database")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    st.header("🧾 Table & Parameters")
    input_table = st.text_input("Input Table Name")
    output_table = st.text_input("Output Table Name")
    key_column = st.text_input("Key Column(s) (comma-separated)")
    groupby_column = st.text_input("GroupBy Column(s) (comma-separated, optional)", "")
    threshold = st.slider("Match Score Threshold", 0.7, 1.0, 0.85, 0.01)

    custom_stopwords = st.text_area("Additional Stopwords (comma-separated)", "").strip()
    submitted = st.form_submit_button("Run")

if submitted:
    try:
        st.info("🔌 Connecting to SQL Server...")
        start = time.time()
        cnxn = get_connection(server, db, user, pwd)
        cursor = cnxn.cursor()
        st.success("✅ Connected!")

        df = pd.read_sql(f"SELECT * FROM {input_table}", cnxn)
        keys = [k.strip() for k in key_column.split(",")]
        df["_key"] = df[keys].astype(str).agg(" ".join, axis=1).fillna("")

        stopwords = DEFAULT_STOPWORDS.union({s.strip().lower() for s in custom_stopwords.split(",") if s.strip()})
        df["_processed"] = df["_key"].apply(lambda x: preprocess_text(x, stopwords))

        group_col = "GroupID"
        score_col = "MatchScore"
        unique_col = "IsUnique"
        all_groups = []
        gid_counter = count(start=1)

        if groupby_column.strip():
            group_cols = [col.strip() for col in groupby_column.split(",")]
            st.info(f"🧠 Grouping by {group_cols}...")

            def process_group(args):
                val, group_df = args
                emb = model.encode(group_df["_processed"].tolist(), convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
                local_gids, scores = group_by_similarity(emb, threshold)
                gid_map = {}
                adjusted_ids = []
                for gid in local_gids:
                    if gid not in gid_map:
                        gid_map[gid] = next(gid_counter)
                    adjusted_ids.append(gid_map[gid])
                group_df[group_col] = adjusted_ids
                group_df[score_col] = scores
                return group_df

            group_data = list(df.groupby(group_cols))
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_group, group_data))
                all_groups.extend(results)
        else:
            st.info("🧠 Processing all records together...")
            emb = model.encode(df["_processed"].tolist(), convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
            gids, scores = group_by_similarity(emb, threshold)
            df[group_col] = gids
            df[score_col] = scores
            all_groups.append(df)

        final_df = pd.concat(all_groups, ignore_index=True)
        final_df[unique_col] = final_df.groupby(group_col)[group_col].transform('count') == 1
        final_df[unique_col] = final_df[unique_col].astype(int)

        st.success("✅ Deduplication complete")
        st.dataframe(final_df.head(30))

        st.info("📤 Writing output to SQL...")
        write_to_sql(final_df.drop(columns=["_key", "_processed"]), output_table, cursor, cnxn, group_col, score_col, unique_col)
        st.success(f"🏁 Done! Time taken: {time.time() - start:.2f}s")

        cursor.close()
        cnxn.close()
    except Exception as e:
        st.error(f"❌ Error: {e}")
