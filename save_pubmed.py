import datasets

pubmed = datasets.load_dataset("pubmed", cache_dir="/data", streaming=True)

save_text = []
save_idx = 1
for idx, entry in enumerate(pubmed['train']):
    
    abstract_text = entry['MedlineCitation']['Article']['Abstract']['AbstractText']
    if abstract_text:
        save_text.append(abstract_text)
        if len(save_text) == 500000:
            with open(f"/data/pubmed/pubmed_{save_idx}", 'w', encoding='utf-8') as f:
                f.write('\n'.join(save_text))
            print(f"Saved pubmed_{save_idx}")
            save_text = []
            save_idx += 1