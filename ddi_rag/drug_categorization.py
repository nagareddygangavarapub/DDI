"""
drug_categorization.py — Drug-name → route and product-type mapping.

Public API:
    lookup_route(drug_name)   -> (route: str | None, method: str)
    categorize_drug(name)     -> category: str | None
    apply_route_column(df)    -> df with openfda_route filled
    apply_product_type(df)    -> df with openfda_product_type filled
"""

import re
import pandas as pd
from difflib import get_close_matches
from config import FUZZY_CUTOFF, MIN_ROOT_LEN

# ── Route lists ───────────────────────────────────────────────────────────────
oral_meds = [
    'amoxicillin and clavulanate potassium','buspirone hydrochloride',
    'olmesartan medoxomil / amlodipine besylate / hydrochlorothiazide',
    'linagliptin and metformin hydrochloride','sacubitril and valsartan',
    'losartan potassium and hydrochlorothiazide','triamterene and hydrochlorothiazide',
    'acetaminophen and codeine phosphate','hydrocodone bitartrate and acetaminophen',
    'lurasidone hydrochloride','omeprazole sodium bicarbonate','febuxostat',
    'allopurinol','fluconazole','pregabalin','carbamazepine','icosapent ethyl',
    'fluoxetine','famotidine','potassium citrate','esomeprazole magnesium',
    'methimazole','clonidine hydrochloride','brivaracetam','metformin hydrochloride',
    'venlafaxine hydrochloride','bumetanide','celecoxib','sertraline hydrochloride',
    'quetiapine','atorvastatin calcium','sildenafil','tamsulosin hydrochloride',
    'propranolol hydrochloride','atenolol','finasteride','olanzapine',
    'phentermine hydrochloride','guanfacine','tizanidine hydrochloride',
    'hydroxychloroquine sulfate','metronidazole','alprazolam','fenofibrate',
    'oxybutynin','bisoprolol fumarate','nortriptyline hydrochloride',
    'metoprolol succinate','lisinopril','ibuprofen','solifenacin succinate',
    'spironolactone','fluvoxamine maleate','benazepril hydrochloride','rufinamide',
    'lithium','lithium carbonate','lithium citrate',
    'gabapentin','probenecid','phenytoin','phenytoin sodium',
    'rifampin','warfarin','warfarin sodium','thyroid',
    'azithromycin','simvastatin','ondansetron','ondansetron hydrochloride',
    'topiramate','amoxicillin','cholestyramine','cyclosporine',
    'digoxin','ciprofloxacin','ciprofloxacin hydrochloride',
    'methocarbamol','phenobarbital','omeprazole','sucralfate',
    'erythromycin','niacin','acyclovir','naproxen','naproxen sodium',
    'lamotrigine','diltiazem','diltiazem hydrochloride',
    'bupropion','bupropion hydrochloride','levothyroxine','levothyroxine sodium',
    'aripiprazole','duloxetine','duloxetine hydrochloride',
    'mirtazapine','lorazepam','cimetidine','ritonavir','atazanavir',
    'emtricitabine and tenofovir disoproxil fumarate',
    'diclofenac','diclofenac sodium','diclofenac potassium',
    'furosemide','metformin hydrochloride tablets',
    'dextroamphetamine sulfate','methylphenidate','methylphenidate hydrochloride',
    'clonazepam','diazepam','valproic acid','valproate sodium','divalproex sodium',
    'levetiracetam','oxcarbazepine','risperidone',
    'ziprasidone','ziprasidone hydrochloride','clozapine','haloperidol',
    'amlodipine','amlodipine besylate','ramipril','enalapril','captopril',
    'valsartan','irbesartan','candesartan','telmisartan','hydrochlorothiazide',
    'chlorthalidone','pravastatin','rosuvastatin','lovastatin','fluvastatin',
    'pantoprazole','lansoprazole','rabeprazole','ranitidine',
    'metoclopramide','loperamide','lactulose','docusate','senna',
    'prednisone','prednisolone','methylprednisolone','dexamethasone',
    'sulfasalazine','hydroxyzine','hydroxyzine hydrochloride',
    'cetirizine','loratadine','fexofenadine','montelukast','clopidogrel','aspirin',
    'tetracycline','doxycycline','minocycline',
    'trimethoprim and sulfamethoxazole','sulfamethoxazole and trimethoprim',
    'nitrofurantoin','clindamycin','linezolid','voriconazole','itraconazole',
    'glyburide','glipizide','glimepiride','pioglitazone','sitagliptin',
    'empagliflozin','canagliflozin','dapagliflozin',
    'estradiol','conjugated estrogens','medroxyprogesterone','testosterone',
    'alendronate','risedronate','colchicine',
    'tacrolimus','mycophenolate mofetil','azathioprine','methotrexate',
    'oxycodone','oxycodone hydrochloride','morphine','morphine sulfate',
    'tramadol','tramadol hydrochloride',
    'amitriptyline','amitriptyline hydrochloride','imipramine',
    'citalopram','escitalopram','paroxetine','trazodone',
    'carvedilol','labetalol','hydralazine',
    'isosorbide mononitrate','isosorbide dinitrate',
    'folic acid','cyanocobalamin','thiamine hydrochloride',
    'pyridoxine hydrochloride','ascorbic acid','phytonadione',
    'water','sterile water',
    'imatinib','imatinib mesylate','dasatinib','nilotinib','bosutinib',
    'erlotinib','erlotinib hydrochloride','gefitinib','afatinib','lapatinib',
    'osimertinib','dabrafenib','vemurafenib','encorafenib','trametinib',
    'sorafenib','sunitinib malate','pazopanib','regorafenib','cabozantinib',
    'axitinib','lenvatinib','palbociclib','ribociclib','abemaciclib',
    'venetoclax','thalidomide','lenalidomide','pomalidomide',
    'everolimus','everolimus tablets','sirolimus',
    'hydroxyurea','mercaptopurine','thioguanine','capecitabine',
    'ibrutinib','acalabrutinib','zanubrutinib','baricitinib','tofacitinib',
    'upadacitinib','apremilast','leflunomide',
    'isoniazid','ethambutol hydrochloride','pyrazinamide','rifabutin','rifapentine',
    'cefdinir','cefuroxime axetil','cephalexin','cefaclor',
    'doxycycline','minocycline',
    'lamivudine','zidovudine','abacavir sulfate','tenofovir disoproxil fumarate',
    'dolutegravir sodium','raltegravir','efavirenz','nevirapine','rilpivirine',
    'darunavir','lopinavir and ritonavir','atazanavir and cobicistat',
    'sumatriptan','sumatriptan succinate','rizatriptan','zolmitriptan',
    'naratriptan','almotriptan','eletriptan hydrobromide','frovatriptan succinate',
    'lasmiditan','ubrogepant','rimegepant sulfate','atogepant',
    'acamprosate calcium','varenicline','varenicline tartrate',
    'clonazepam','diazepam','alprazolam','lorazepam','triazolam',
    'zaleplon','ramelteon','suvorexant','lemborexant',
    'carbidopa and levodopa','carbidopa levodopa','levodopa','carbidopa',
    'ropinirole','ropinirole hydrochloride','pramipexole',
    'amantadine','amantadine hydrochloride',
    'valbenazine','deutetrabenazine','tetrabenazine',
    'fingolimod','fingolimod hydrochloride','siponimod','teriflunomide',
    'buprenorphine','buprenorphine and naloxone','methadone','methadone hydrochloride',
    'naltrexone','naltrexone hydrochloride','naloxegol oxalate',
    'oral semaglutide','liraglutide','semaglutide','dulaglutide','exenatide',
    'ivacaftor','elexacaftor, tezacaftor, and ivacaftor','lumacaftor and ivacaftor',
    'amphetamine','methylphenidate',
    'serdexmethylphenidate and dexmethylphenidate','viloxazine hydrochloride',
]

injectable_meds = [
    'piperacillin and tazobactam','calcium gluconate','nicardipine hydrochloride',
    'bupivacaine hydrochloride','cyclophosphamide','amiodarone hydrochloride',
    'zoledronic acid','potassium phosphates','calcium chloride','epinephrine',
    'carboplatin','naloxone hydrochloride','ethacrynate sodium','oxaliplatin',
    'levofloxacin','ropivacaine hydrochloride','vasopressin','bivalirudin',
    'fulvestrant','caspofungin acetate','bortezomib','cefazolin','irinotecan',
    'dexmedetomidine','daptomycin','thiotepa','dobutamine','arsenic trioxide',
    'amphotericin b','vancomycin','vancomycin hydrochloride',
    'heparin','heparin sodium','enoxaparin',
    'insulin','insulin lispro','insulin glargine','insulin aspart',
    'insulin regular','insulin detemir',
    'dexamethasone injection','methylprednisolone sodium succinate',
    'labetalol hydrochloride','hydralazine hydrochloride',
    'phenylephrine','norepinephrine','dopamine','adenosine','atropine',
    'magnesium sulfate','sodium bicarbonate','potassium chloride',
    'rituximab','trastuzumab','pertuzumab','bevacizumab','cetuximab',
    'paclitaxel','docetaxel','gemcitabine','cisplatin','fluorouracil','leucovorin',
    'vincristine','azacitidine','decitabine',
    'midazolam','propofol','fentanyl','hydromorphone','ketorolac',
    'succinylcholine','succinylcholine chloride',
    'meropenem','methylene blue','esmolol hydrochloride',
    'oxytocin','flumazenil','terbutaline sulfate','argatroban','eptifibatide',
    'pemetrexed disodium','busulfan','cefoxitin',
    'nelarabine','sodium nitroprusside','topotecan','fludarabine phosphate','ifosfamide',
    'acetylcysteine','cytarabine','granisetron hydrochloride','cladribine',
    'carmustine','bendamustine hydrochloride','ertapenem sodium',
    'methylergonovine maleate','fosaprepitant dimeglumine','abatacept',
    'ampicillin sodium and sulbactam sodium','pamidronate disodium',
    'cefepime','mannitol','vecuronium bromide','milrinone lactate',
    'fondaparinux sodium','romidepsin','caffeine citrate',
    'amikacin','penicillin g potassium','daunorubicin hydrochloride',
    'tocilizumab','mitoxantrone','bleomycin','bleomycin sulfate',
    'alteplase','aldesleukin','albumin human',
    'dalteparin sodium','tenecteplase','ferumoxytol','epirubicin hydrochloride',
    'thiamine hydrochloride','ceftriaxone','remifentanil hydrochloride',
    'iron sucrose','cefotaxime','ustekinumab',
    'fomepizole','etoposide','etoposide phosphate',
    'tobramycin','tobramycin sulfate',
    'ocrelizumab','natalizumab','mepolizumab',
    'adalimumab','adalimumab-ryvk','adalimumab-bwwd','adalimumab-atto',
    'eculizumab','omalizumab','ranibizumab','somatropin',
    'glucagon','glucagon injection',
    'golimumab','goserelin','triptorelin','etanercept',
    'infliximab','vedolizumab','basiliximab','secukinumab','ixekizumab',
    'guselkumab','risankizumab','dupilumab','tralokinumab',
    'anakinra','peginterferon alfa-2a',
    'penicillin g benzathine','penicillin g sodium','penicillin g',
    'micafungin','micafungin sodium',
    'paliperidone palmitate','sugammadex','deferoxamine mesylate','treprostinil',
    'iron dextran','tirzepatide','clevidipine',
    'daratumumab','rituximab','ocrelizumab',
    'doxorubicin','doxorubicin hydrochloride',
    'melphalan','chlorambucil','mechlorethamine hydrochloride',
    'vinblastine sulfate','cabazitaxel',
    'polatuzumab vedotin','brentuximab vedotin','sacituzumab govitecan',
    'ado-trastuzumab emtansine','mirvetuximab soravtansine',
    'blinatumomab','dinutuximab',
    'nalbuphine hydrochloride','butorphanol tartrate',
    'elapegademase-lvlr','pegaspargase','pralatrexate',
    'iohexol','gadodiamide','gadoterate meglumine',
    'sodium fluoride','dextrose','sterile water for injection','water injection',
    'magnesium sulfate heptahydrate','soybean oil',
    'leuprolide acetate',
    'aztreonam','imipenem and cilastatin','ertapenem','tigecycline',
    'oritavancin','telavancin hydrochloride','foscarnet sodium','ganciclovir sodium',
    'mesna','dexrazoxane','oxytocin','methylergonovine maleate',
    'follitropin','menotropins','pramlintide acetate',
    'immune globulin intravenous (human) %','nirsevimab','palivizumab',
    'adrenalin (epinephrine)',
]

topical_meds = [
    'lidocaine','diclofenac sodium gel','mupirocin','mometasone furoate',
    'clindamycin phosphate and benzoyl peroxide','tazarotene','adapalene',
    'clotrimazole','hydroquinone','ketoconazole','tretinoin','fentanyl transdermal',
    'scopolamine transdermal','brimonidine tartrate','latanoprost','timolol maleate',
    'tacrolimus ointment','pimecrolimus','clobetasol propionate','betamethasone',
    'triamcinolone acetonide','hydrocortisone','metronidazole gel',
    'ivermectin topical','permethrin','nystatin','miconazole',
    'terbinafine','salicylic acid','benzoyl peroxide',
    'testosterone gel','estradiol patch','nitroglycerin ointment',
    'dorzolamide','travoprost','bimatoprost',
    'sulfacetamide sodium','sulfacetamide sodium and prednisolone sodium phosphate',
    'moxifloxacin ophthalmic','moxifloxacin ophthalmic solution',
    'tropicamide','apraclonidine','cyclopentolate hydrochloride',
    'dorzolamide hydrochloride timolol maleate',
    'crotamiton','oxiconazole nitrate','penciclovir','capsaicin',
    'methoxsalen','dapsone gel, %','sofpironium bromide','trifarotene','rotigotine',
    'efinaconazole','ofloxacin otic',
]

inhaled_meds = [
    'albuterol sulfate','fluticasone propionate and salmeterol',
    'budesonide inhalation','ipratropium bromide','arformoterol tartrate',
    'tiotropium bromide','formoterol fumarate',
    'beclomethasone dipropionate','ciclesonide','levalbuterol',
    'umeclidinium','aclidinium bromide','indacaterol','vilanterol','salmeterol',
    'budesonide and formoterol','umeclidinium and vilanterol',
    'fluticasone furoate and vilanterol',
    'glycopyrrolate inhalation','glycopyrrolate and formoterol fumarate',
    'glycopyrronium','olodaterol respimat inhalation spray',
    'mometasone furoate inhalation','fluticasone propionate','revefenacin',
]

vaginal_meds = [
    'etonogestrel and ethinyl estradiol','progesterone vaginal',
    'clindamycin vaginal','metronidazole vaginal',
    'miconazole vaginal','terconazole','boric acid vaginal',
    'dinoprostone','alprostadil',
]

nasal_meds = [
    'azelastine hydrochloride','fluticasone propionate nasal',
    'mometasone furoate nasal','triamcinolone acetonide nasal',
    'budesonide nasal','oxymetazoline','ipratropium bromide nasal',
    'ciclesonide nasal','olopatadine hydrochloride and mometasone furoate',
    'zavegepant',
]

# ── Build route_map ───────────────────────────────────────────────────────────
route_map: dict = {}
for _d in oral_meds:       route_map[_d.lower().strip()] = "oral"
for _d in injectable_meds: route_map[_d.lower().strip()] = "intravenous"
for _d in topical_meds:    route_map[_d.lower().strip()] = "topical"
for _d in inhaled_meds:    route_map[_d.lower().strip()] = "respiratory (inhalation)"
for _d in vaginal_meds:    route_map[_d.lower().strip()] = "vaginal"
for _d in nasal_meds:      route_map[_d.lower().strip()] = "nasal"

# ── Salt-suffix normalization ─────────────────────────────────────────────────
_SALT_SUFFIXES = [
    r"\bhydrochloride\b", r"\bhcl\b", r"\bsulfate\b", r"\bsodium\b",
    r"\bpotassium\b", r"\bcalcium\b", r"\bacetate\b", r"\bmaleate\b",
    r"\bfumarate\b", r"\bsuccinate\b", r"\bbesylate\b", r"\btartrate\b",
    r"\bmesylate\b", r"\bphosphate\b", r"\bgluconate\b", r"\bbromide\b",
    r"\bbitartrate\b", r"\bmedoxomil\b", r"\bmagnesium\b", r"\bmonobasic\b",
    r"\bdibasic\b", r"\bsaccharate\b", r"\baspartate\b",
]


def _normalize(name: str) -> str:
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    for p in _SALT_SUFFIXES:
        name = re.sub(p, "", name)
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


# Pre-build normalized reference
_normalized_ref: dict = {}
for _drug, _route in route_map.items():
    _nk = _normalize(_drug)
    if _nk not in _normalized_ref:
        _normalized_ref[_nk] = _route

_ref_keys = list(_normalized_ref.keys())


def lookup_route(drug_name) -> tuple:
    """
    Try 4 layers to map a drug name to a route of administration.

    Returns:
        (route: str | None, method: str)
    """
    if pd.isna(drug_name) or str(drug_name).strip() == "":
        return None, "empty"

    raw = str(drug_name).lower().strip()

    # Layer 1 — Exact match
    if raw in route_map:
        return route_map[raw], "exact"

    # Layer 2 — Normalized match (removes salt suffixes)
    norm = _normalize(raw)
    if norm in _normalized_ref:
        return _normalized_ref[norm], "normalized"

    # Layer 3 — First-word (root) match
    words = norm.split()
    root  = words[0] if words else ""
    if len(root) >= MIN_ROOT_LEN:
        candidates = [
            (rk, r) for rk, r in _normalized_ref.items()
            if rk.split() and rk.split()[0] == root
        ]
        if len(candidates) == 1:
            return candidates[0][1], "root_match"
        elif len(candidates) > 1:
            routes = list(set(r for _, r in candidates))
            if len(routes) == 1:
                return routes[0], "root_match_safe"
            return "needs_review", f"ambiguous_root:{root}"

    # Layer 4 — Fuzzy fallback
    matches = get_close_matches(norm, _ref_keys, n=1, cutoff=FUZZY_CUTOFF)
    if matches:
        return _normalized_ref[matches[0]], "fuzzy"

    return "needs_review", "unmatched"


def apply_route_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing openfda_route values using lookup_route().
    Returns the DataFrame with openfda_route partially or fully filled.
    """
    if "openfda_route" not in df.columns:
        df["openfda_route"] = None

    mask = df["openfda_route"].isna()
    results = df.loc[mask, "final_generic_name"].apply(
        lambda x: lookup_route(x)[0]
    )
    df.loc[mask, "openfda_route"] = results
    print(
        f"Route fill: {mask.sum()} rows processed. "
        f"Remaining NaN: {df['openfda_route'].isna().sum()}"
    )
    return df


# ── Product-type categorisation ───────────────────────────────────────────────
cellular_therapy: set = {
    "tisagenlecleucel", "lisocabtagene maraleucel", "idecabtagene vicleucel",
    "ciltacabtagene autoleucel", "afamitresgene autoleucel",
    "betibeglogene autotemcel", "atidarsagene autotemcel",
    "elivaldogene autotemcel", "lovotibeglogene autotemcel",
    "onasemnogene abeparvovec-xioi", "onasemnogene abeparvovec-brve",
    "delandistrogene moxeparvovec-rokl", "valoctocogene roxaparvovec-rvox",
    "exagamglogene autotemcel",
}

human_otc_drugs: set = {
    "acetaminophen", "ibuprofen", "naproxen", "naproxen sodium", "aspirin",
    "cetirizine", "loratadine", "fexofenadine", "diphenhydramine hydrochloride",
    "famotidine", "omeprazole", "esomeprazole magnesium", "lansoprazole",
    "loperamide", "docusate", "senna", "melatonin",
}

_all_drugs = (
    set(d.strip().lower() for d in oral_meds)
    | set(d.strip().lower() for d in injectable_meds)
    | set(d.strip().lower() for d in topical_meds)
    | set(d.strip().lower() for d in inhaled_meds)
    | set(d.strip().lower() for d in vaginal_meds)
    | set(d.strip().lower() for d in nasal_meds)
)

human_prescription_drugs: set = _all_drugs - cellular_therapy - human_otc_drugs

_cat_map = (
    {d: "cellular_therapy"        for d in cellular_therapy}
    | {d: "human_otc_drug"        for d in human_otc_drugs}
    | {d: "human_prescription_drug" for d in human_prescription_drugs}
)


def categorize_drug(drug_name) -> str | None:
    """
    Classify a drug name into: cellular_therapy, human_otc_drug,
    human_prescription_drug, or None (via fuzzy fallback).
    """
    if pd.isna(drug_name):
        return None
    name = str(drug_name).strip().lower()
    if name in _cat_map:
        return _cat_map[name]

    # Fuzzy fallback
    close = get_close_matches(name, _cat_map.keys(), n=1, cutoff=0.85)
    return _cat_map[close[0]] if close else None


def apply_product_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing openfda_product_type values using categorize_drug().
    Remaining unresolved rows are labeled 'no data from<fda>'.
    """
    if "openfda_product_type" in df.columns:
        df["openfda_product_type"] = df["openfda_product_type"].str.replace(
            " ", "_", regex=False
        )

    mask = df["openfda_product_type"].isna()
    df.loc[mask, "openfda_product_type"] = (
        df.loc[mask, "final_generic_name"]
        .apply(lambda x: categorize_drug(x) if pd.notna(x) else "no data from<fda>")
        .fillna("no data from<fda>")
    )
    print("Product type value counts:")
    print(df["openfda_product_type"].value_counts(dropna=False))
    return df
