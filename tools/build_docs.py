"""
build_docs.py - turns the Axon/EDC JSON files in ../data into vector_docs_new.jsonl
-----------------------------------------------------------------------------------
Run:   python tools/build_docs.py          (from the meta-poc folder, venv active)
Then:  rename vector_docs_new.jsonl to vector_docs.jsonl, delete the chroma_db folder, restart the app.

This is the file to adapt when the REAL Axon / EDC API data arrives:
  1. save the API responses as JSON files in ../data (or load them directly in load())
  2. adjust the field names inside the make_* functions
  3. keep the output format: {"id": ..., "type": ..., "text": ..., "metadata": {...}}

GOLDEN RULE: the "text" of each document must be a complete sentence-style description that makes
sense on its own, because the vector search and the LLM only see this text.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "vector_docs_new.jsonl")


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def names(items, key="name"):
    """['a','b'] from [{'name':'a'}, {'name':'b'}]"""
    return ", ".join(i[key] for i in items) or "none"


def fact(item, attribute):
    """Read one value from an EDC 'facts' list."""
    for f in item["facts"]:
        if f["attributeId"] == attribute:
            return f["value"]
    return None


def doc(id_, type_, text, **metadata):
    return {"id": id_, "type": type_, "text": text, "metadata": metadata}


# ---------------------------------------------------------------- Axon objects
def make_term(g):
    text = (f"Glossary term '{g['name']}' (domain {g['domain']}, status {g['status']}). Definition: {g['definition']} "
            f"Steward: {g['steward']['name']} ({g['steward']['dept']}). Owner: {g['owner']['name']}. "
            f"Classification: {g['classification']} {', '.join(g['tags'])}. Critical data element: {g['critical_data_element']}. "
            f"Linked KPIs: {names(g['linked_kpis'])}. "
            f"Physical columns: {names(g['linked_physical_columns'][:8])}. Systems: {', '.join(g['linked_systems'])}. "
            f"Average DQ score {g['avg_dq_score']}%.")
    return doc(g["id"], "glossary_term", text, name=g["name"], domain=g["domain"], classification=g["classification"], source="Axon")


def make_kpi(k):
    text = (f"KPI '{k['name']}' ({k['id']}): {k['description']} Formula: {k['formula']}. Target {k['target']}, "
            f"frequency {k['frequency']}. Owner: {k['owner']['name']} ({k['owner']['dept']}). "
            f"Glossary terms: {names(k['linked_glossary_terms'])}. Mart columns: {', '.join(k['linked_mart_columns'])}. "
            f"Reports using it: {names(k['linked_reports'])}. Source systems: {', '.join(k['source_systems'])}. "
            f"DQ score {k['avg_dq_score']}%.")
    return doc(k["id"], "kpi", text, name=k["name"], source="Axon")


def make_report(p):
    text = (f"Report '{p['name']}' ({p['report_type']}, id {p['id']}), refreshed {p['refresh_frequency']}. "
            f"Owner: {p['report_owner']['name']} ({p['report_owner']['dept']}). "
            f"Classification: {p['classification']} {', '.join(p['tags'])}. Criticality: {p['criticality']}. "
            f"KPIs: {names(p['linked_kpis'])}. Glossary terms: {names(p['linked_glossary_terms'])}. "
            f"Attributes: {', '.join(a['mart_column'] for a in p['linked_attributes'])}. "
            f"Source systems: {', '.join(p['source_systems'])}. Average DQ score {p['avg_dq_score']}%. "
            f"Monthly users: {p['monthly_users']}.")
    return doc(p["id"], "report", text, name=p["name"], classification=p["classification"],
               owner=p["report_owner"]["name"], source="Axon+EDC")


def make_system(s):
    text = (f"System '{s['name']}': {s['description']} Technology {s['technology']}, criticality {s['criticality']}. "
            f"System owner: {s['system_owner']['name']} ({s['system_owner']['dept']}, {s['system_owner']['email']}). "
            f"Technical owner: {s['technical_owner']['name']}. Scanned in EDC: {s['scanned_in_edc']} ({s['table_count']} tables).")
    return doc(s["id"], "system", text, name=s["name"], source="Axon")


# ---------------------------------------------------------------- EDC objects
def make_table(t):
    text = (f"Table {t['schema']}.{t['name']} in {t['layer']} layer, system {t['system']}. {t['row_count']:,} rows, "
            f"{t['column_count']} columns: {', '.join(t['columns'])}. Classification {t['classification']} "
            f"{', '.join(t['tags'])}. DQ score {t['dq_score']}%. System owner: {t['system_owner']}. "
            f"Glossary terms: {', '.join(t['glossary_terms'][:10])}.")
    return doc("TBL-" + t["id"], "table", text, name=t["name"], layer=t["layer"], source="EDC")


def make_column(item, cls, dq, impact, id_to_name):
    """One document per column. Combines EDC facts + classification + DQ + lineage + impact."""
    name = id_to_name[item["id"]]
    term = fact(item, "com.infa.ldm.axon.associatedGlossary")
    parents = [id_to_name.get(l["id"], l["id"]) for l in item["srcLinks"]]
    text = (f"Column {name}, datatype {fact(item, 'com.infa.ldm.relational.Datatype')}, resource {fact(item, 'core.resourceName')}. "
            f"Business term: {term or 'NOT LINKED to glossary'}. "
            f"Classification {cls.get('level')} {', '.join(cls.get('tags', []))}. DQ score {dq}%. "
            f"Direct upstream columns: {', '.join(parents) if parents else 'none (source column)'}. ")
    if impact:
        text += f"Impact: {impact['summary']}"
    return doc("COL-" + item["id"], "column", text, column=name, classification=cls.get("level", ""), source="EDC")


def make_lineage(path_item, n, path):
    text = f"Lineage of {path_item['target_name']} ({path_item['layer']}): " + " -> ".join(path)
    return doc(f"LIN-{path_item['target']}-{n}", "lineage", text, target=path_item["target_name"], source="EDC")


def make_impact(x):
    text = (f"Impact analysis for {x['column']}: {x['summary']} Impact level {x['impact_level']}. "
            f"Impacted reports: {names(x['impacted_reports'])}. Impacted KPIs: {names(x['impacted_kpis'])}.")
    return doc("IMP-" + x["column_id"], "impact", text, column=x["column"], impact_level=x["impact_level"], source="EDC")


# ---------------------------------------------------------------- main
def main():
    docs = []
    docs += [make_term(g) for g in load("axon_glossary_terms.json")["items"]]
    docs += [make_kpi(k) for k in load("axon_kpis.json")["items"]]
    docs += [make_report(p) for p in load("axon_products_reports.json")["items"]]
    docs += [make_system(s) for s in load("axon_systems.json")["items"]]
    docs += [make_table(t) for t in load("edc_tables.json")["items"]]

    cls_by_id = {c["column_id"]: c for c in load("classification.json")["items"]}
    dq_by_id = {d["column_id"]: d["overall_score"] for d in load("dq_scores.json")["items"]}
    impact_items = load("impact_analysis.json")["items"]
    impact_by_id = {x["column_id"]: x for x in impact_items}

    columns = load("edc_columns.json")["items"]
    id_to_name = {c["id"]: fact(c, "core.location").strip("/").replace("/", ".") for c in columns}
    for c in columns:
        docs.append(make_column(c, cls_by_id.get(c["id"], {}), dq_by_id.get(c["id"]), impact_by_id.get(c["id"]), id_to_name))

    for item in load("edc_lineage_paths.json")["items"]:
        for n, path in enumerate(item["paths"][:3]):
            docs.append(make_lineage(item, n, path))
    docs += [make_impact(x) for x in impact_items if x["layer"] in ("SOURCE", "DWH")]

    with open(OUT, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    counts = {}
    for d in docs:
        counts[d["type"]] = counts.get(d["type"], 0) + 1
    print("written:", OUT)
    print("documents:", len(docs), counts)
    assert len({d["id"] for d in docs}) == len(docs), "duplicate ids!"


if __name__ == "__main__":
    main()
