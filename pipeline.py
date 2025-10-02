#%%
import pandas as pd
import re
import numpy as np

try:
    zh_songs = pd.read_pickle("./zh_songs_v2.pkl")
    print(f"chinese songs: {len(zh_songs)}")
except FileNotFoundError:
    print("creating the dataframe")
    #%%
    suno_df = pd.read_pickle("./suno_metadata_202509.pkl")
    suno_df['source'] = 'suno'
    suno_df['url'] = 'https://suno.com/song/' + suno_df['id'].astype(str)
    suno_df.rename(columns={'prompt': 'lyrics', 'gpt_description_prompt':'prompt'}, inplace=True)

    udio_df = pd.read_pickle("./udio_metadata_202509.pkl")
    udio_df['source'] = 'udio'
    udio_df['url'] = 'https://udio.com/songs/' + udio_df['id'].astype(str)

    df = pd.concat([suno_df, udio_df], ignore_index=True)
    df = df[['id', 'source', 'url', 'title','lyrics', 'prompt','tags']]
    # replace Null lyrics with ""
    df['lyrics'] = df['lyrics'].fillna("")

    print(f"Suno: {len(suno_df)}\nUdio: {len(udio_df)}\nTOTAL:{len(df)}")

    #%% regex
    print("Regex cleaning")
    # remove any part between parentheses in the lyrics
    df['lyrics'] = df['lyrics'].apply(lambda x: re.sub(r'\[.*?\]', '', x))
    df['lyrics'] = df['lyrics'].apply(lambda x: re.sub(r'\(.*?\)', '', x))
    df['lyrics'] = df['lyrics'].apply(lambda x: re.sub(r'\{.*?\}', '', x))
    # also use （ ）characters
    df['lyrics'] = df['lyrics'].apply(lambda x: re.sub(r'（.*?）', '', x))
    # remove any of Chorus:, Verse:, Pre-Chorus:
    df['lyrics'] = df['lyrics'].apply(lambda x: re.sub(r'\b(Chorus|Verse|Pre-Chorus|Bridge|Outro|Intro):\s*', '', x, flags=re.IGNORECASE))
    # remove urls
    df['lyrics'] = df['lyrics'].apply(lambda x: re.sub(r'http[s]?://\S+', '', x))
    # remove any characters that is not chinese
    # df['lyrics'] = df['lyrics'].apply(lambda x: re.sub(r'[^\u4e00-\u9fa5\s]', '', x))
    # replace any whitespace and newlines with a single space
    df['lyrics'] = df['lyrics'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    #%% detect language
    #apt install icu and then pip install pyicu and pycld2; see docs
    print("language detection")
    from polyglot.detect import Detector
    from tqdm.auto import tqdm
    tqdm.pandas()
    # mute warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='polyglot')

    def detect_language_polyglot(lyrics):
        # Suppress stdout temporarily
        class DummyFile(object):
            def write(self, x): pass
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        try:
            detector = Detector(lyrics, quiet=True)
            result = (detector.language.name, detector.language.code, detector.language.confidence)
        except Exception:
            result = (None, None, -1)
        finally:
            sys.stdout = save_stdout
        return result
        
    df[['language', 'lang_code', 'language_prob']] = df["lyrics"].progress_apply(lambda x: detect_language_polyglot(x)).tolist()
    # chinese_iso = ['yue_Hant', 'zho_Hans', "cdo_","cjy_","cmn_","cnp_","cpx_","csp_","czh_","czo_","gan_","hak_","hnm_","hsn_","luh_","lzh_","mnp_","nan_","sjc_","wuu_","yue_",]

    def find_chinese(df):
        if df['language'] == None: return False
        elif df['language'].startswith("Chin"): return True
        else: return False

    df['is_chinese'] = df.progress_apply(find_chinese, axis=1)

    zh_songs = df[df['is_chinese'] == True].copy()
    zh_songs = zh_songs.drop_duplicates(subset=['lyrics'])
    zh_songs.to_pickle("zh_songs_v2.pkl")
    # zh_songs.sort_values('language_prob')

#%% translate lyrics
# import asyncio
# from googletrans import Translator

# async def translate_bulk():
#     async with Translator() as translator:
#         lyrics_list = zh_songs['lyrics'].tolist()
#         translations = await translator.translate(
#             lyrics_list, 
#             dest="en"
#         )
#         zh_songs['translated_lyrics'] = [translation.text for translation in translations]
#         # for translation in translations:
#             # print(translation.origin, " -> ", translation.text)

# if "translated_lyrics" not in zh_songs.columns:
#     print("translating")
#     asyncio.run(translate_bulk())
#     zh_songs.to_pickle("./zh_songs_v2.pkl")

#%% embeddings
if "emb" not in zh_songs.columns:
    print("computing embeddings")
    # import torch.nn.functional as F
    from torch import Tensor
    from transformers import AutoTokenizer, AutoModel
    import torch
    from tqdm import tqdm

    input_texts = zh_songs['lyrics'].tolist()
    print(f"Number of input texts: {len(input_texts)}")
    model_name = 'Qwen/Qwen3-Embedding-8B' 
    # tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large-zh")
    # model = AutoModel.from_pretrained("thenlper/gte-large-zh")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side='left'
        )
    model = AutoModel.from_pretrained(
        'Qwen/Qwen3-Embedding-8B',
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # trust_remote_code=True,
        use_cache=True,  # Enable caching for faster inference
        # quantize
    )

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=1024, padding=True, truncation=True, return_tensors='pt').to('cuda')
    # model.to('cuda')
    model.eval()

    # encode inputs in batches of 32
    def encode_batch(batch: Tensor) -> Tensor:
        with torch.no_grad():
            outputs = model(**batch)
        return outputs.last_hidden_state.mean(dim=1)

    # Encode the input texts in batches
    batch_size = 16
    embeddings = []
    for i in tqdm(range(0, len(input_texts), batch_size)):
        batch = {k: v[i:i + batch_size] for k, v in batch_dict.items()}
        embeddings.append(encode_batch(batch))

    # Concatenate all embeddings into a single tensor
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    # scores = (embeddings[:1] @ embeddings[1:].T) * 100

    # print(scores.tolist())
    print(embeddings.shape)
    zh_songs['emb'] = embeddings.cpu().type(torch.float32).numpy().tolist()
    zh_songs.to_pickle("zh_songs_v2.pkl")

#%% umap_hdbscan
if "hdb_label" not in zh_songs.columns:
    import umap
    import numpy as np

    print("umap 5d")
    reducer5D = umap.UMAP(
            n_jobs=32, 
            n_neighbors=15,
            min_dist=0.0,
            n_components=5,
            metric="manhattan",
            # random_state=42, #fixing seed turns off parallelization
        )

    embeddings5D = reducer5D.fit_transform(zh_songs['emb'].apply(np.array).values.tolist())
    print(embeddings5D.shape)
    zh_songs['umap5d'] = embeddings5D.tolist()

    # for component in range(X_embedded.shape[1]):
    #     yiren_prompts[f'umap_d{component}'] = X_embedded[:,component]

    from sklearn.cluster import HDBSCAN

    print("clustering")
    hdb = HDBSCAN(
        n_jobs=32,
        min_samples=5,
        min_cluster_size=10,
        max_cluster_size=200,
        # cluster_selection_method='eom',
        # cluster_selection_epsilon=0.25,
    )

    zh_songs['hdb_label'] = hdb.fit_predict(embeddings5D)
    zh_songs['hdb_label'] = zh_songs['hdb_label'].astype(str)
    zh_songs['hdb_label'].value_counts().sort_values()
    zh_songs.to_pickle("zh_songs_v2.pkl")
#%% naming
if "qwen_hdb_name" not in zh_songs.columns:

    print("bag of word naming")
    hdb_names = {
        'hdb_label': [],
        'hdb_name': [],
        'top_words': [],
    }

    for label in zh_songs['hdb_label'].unique():
        if label == '-1':
            print(f"LABEL {label}: OUTLIERS")
            hdb_names['hdb_label'].append(label)
            hdb_names['hdb_name'].append('OUTLIERS')
            hdb_names['top_words'].append('')
        else:
            lyrics = zh_songs[zh_songs['hdb_label'] == label]['lyrics'].to_list()
            # split on whitespace and count words
            all_words = ' '.join(lyrics).split()
            word_counts = pd.Series(all_words).value_counts()
            name, count = word_counts[:3].index.tolist(), word_counts[:3].values.tolist()
            topk = ' | '.join([f"{n} ({c})" for n, c in zip(name, count)])
            hdb_name = '_'.join([n for n in name if len(n) > 1])
            print(f"LABEL {label}: {topk}")
            hdb_names['hdb_label'].append(label)
            hdb_names['hdb_name'].append(' '.join(hdb_name))
            hdb_names['top_words'].append(topk)

    ## QWEN 3 CLUSTER NAMING
    print("qwen3 naming")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # model_name = "Qwen/Qwen3-4B-Instruct-2507"
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"


    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.bfloat16,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        use_cache=True,  # Enable caching for faster inference
    )

    qwen_hdb_names = {
        'hdb_label': [],
        'qwen_hdb_name': [],
    }

    for cluster in zh_songs['hdb_label'].unique():
        if cluster == '-1':
            print(f"Cluster {cluster}: OUTLIERS")
            qwen_hdb_names['hdb_label'].append(cluster)
            qwen_hdb_names['qwen_hdb_name'].append('OUTLIERS')
        else:
            lyrics = zh_songs[zh_songs['hdb_label']==cluster].lyrics.values

            # prepare the model input
            prompt = """Your going to recieve a list of lyrics in Chinese. 
            These lyrics have been clustered together based on their similarity.
            Output a name for the cluster that captures the thing they have in common.
            E.g. "love songs", "food", "family".
            USE between one and three words.
            """
            messages = [{"role": "user", "content": prompt}] + [{"role": "user", "content": f"{lyric}"} for lyric in lyrics] + [{"role": "user", "content": "name:"}]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # conduct text completion
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            # print("thinking content:", thinking_content)
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            # print("content:", content)
            qwen_hdb_names['hdb_label'].append(cluster)
            qwen_hdb_names['qwen_hdb_name'].append(content)
            print(f"Cluster {cluster}: {content}")

    if "hdb_name" not in zh_songs.columns:
        hdb_names = pd.DataFrame(hdb_names)
        zh_songs = zh_songs.merge(hdb_names, on='hdb_label', how='left')

    # zh_songs.drop(columns=['qwen_hdb_name'], inplace=True)
    if "qwen_hdb_name" not in zh_songs.columns:
        qwen_hdb_names = pd.DataFrame(qwen_hdb_names)
        zh_songs = zh_songs.merge(qwen_hdb_names, on='hdb_label', how='left')

    zh_songs.to_pickle("zh_songs_v2.pkl")

#%% translate the label
# if "translated_qwen_hdb_name" not in zh_songs.columns:
#     import asyncio
#     from googletrans import Translator
#     print("translating qwen label")
#     async def translate_bulk():
#         async with Translator() as translator:
#             lyrics_list = zh_songs['qwen_hdb_name'].tolist()
#             translations = await translator.translate(
#                 lyrics_list, 
#                 dest="en"
#             )
#             zh_songs['translated_qwen_hdb_name'] = [translation.text for translation in translations]
#             # for translation in translations:
#                 # print(translation.origin, " -> ", translation.text)

#     asyncio.run(translate_bulk())
#     zh_songs.to_pickle("zh_songs.pkl")

#%% umap2d
if "umap_x" not in zh_songs.columns:
    import umap.umap_ as umap
    print("umap 2d")
    reducer2D = umap.UMAP(
            n_jobs=40,
            n_components=2,
            n_neighbors=25,
            min_dist=0.5,
            # metric="cosine",
            # random_state=42,
        )

    embeddings2D = reducer2D.fit_transform(
        zh_songs['emb'].apply(np.array).values.tolist(), 
        y=zh_songs['hdb_label']
        )

    zh_songs['umap_x'] = embeddings2D[:,0]
    zh_songs['umap_y'] = embeddings2D[:,1]
    zh_songs.to_pickle('./zh_songs_v2.pkl')

#%% distance 2 centroid
if "dist2centroid" not in zh_songs.columns:
    print("dist2centroid")
    zh_songs[['cluster_centroid_x', 'cluster_centroid_y']] = zh_songs.groupby('hdb_label')[['umap_x', 'umap_y']].transform('mean')
    zh_songs['dist2centroid'] = ((zh_songs['umap_x'] - zh_songs['cluster_centroid_x'])**2 + (zh_songs['umap_y'] - zh_songs['cluster_centroid_y'])**2)**0.5
    zh_songs = zh_songs.sort_values('dist2centroid', ascending=True)
    zh_songs.to_pickle("zh_songs_v2.pkl")


#%% plot
print("plotting")
import plotly.express as px
import plotly.graph_objects as go
import webbrowser
from IPython.display import display

names = 'qwen_hdb_name'

print("plotting")
cluster_names = {k:v for k,v in zh_songs[['hdb_label', names]].values}
new_legend = {name : f"{label}) {name}" for label,name in zh_songs[['hdb_label',names,]].sort_values(names).values}

scatter = px.scatter(
    zh_songs.sort_values('hdb_label', key=lambda x: x.astype(int)), 
    x="umap_x", 
    y="umap_y", 
    color=names,
    color_discrete_map={"-1":"white", "OUTLIERS":"white"},
    # hover_data=["id", "source", "language", "hdb_name","hdb_label","title","qwen_hdb_name", "translated_qwen_hdb_name"], 
    )


fig = go.Figure(layout={'hovermode': 'closest'})
fig.update_layout(width=2000, height=1500)
fig.add_traces(scatter.data)
# fig.update_traces(marker=dict(opacity=0.25))
fig.update_layout(
title="HDBSCAN clusters of prompts embeddings",
)

def do_click(trace, points, state):
    if points.point_inds:
        ind = points.point_inds[0]
        point_id = trace.customdata[ind][0]
        source = trace.customdata[ind][1]
        print(point_id, source)
        if source == 'suno':
            url = "https://www.suno.com/song/" + point_id
        elif source == 'udio':
            url = "https://www.udio.com/songs/" + point_id
        display(url)
        print(url)
        webbrowser.open_new_tab(url)
[s.on_click(do_click) for s in fig.data]



fig.for_each_trace(
     lambda t: t.update(name = new_legend[t.name], legendgroup = new_legend[t.name])
     )
for label in zh_songs['hdb_label'].unique():
    if label == '-1':
        continue
    
    centroid = zh_songs[zh_songs['hdb_label'] == label][['umap_x','umap_y']].median()
    annotation = cluster_names[label]
    legend_name = str(label) +') '+ cluster_names[label]
    
    for t in fig.select_traces({'name': legend_name}):
            centroid = centroid.values
            t.update(
                x = np.append(t.x, centroid[0]),
                y = np.append(t.y, centroid[1]),
                mode='markers+text',
                text = [""]*len(t.x - 1) + [label],
                textfont=dict(
                    family="Arial",
                    size=18,
                    weight='bold',
                    color="black"
                ),
                marker=dict(
                    size=8,
                    color=t.marker.color,  # Use the same color as the cluster
                    line=dict(width=1, color='DarkSlateGrey'),
                    opacity=0.75
                )
                )
# remove margins
fig.update_layout(
    margin=dict(l=0, r=0, t=30, b=0),
    legend_title_text='Cluster',
    legend=dict(
        orientation="v",
        x=1
    ),
)

# save plotly figure
fig.write_image("./zh_songs_v2.png")

