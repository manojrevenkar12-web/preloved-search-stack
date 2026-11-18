"""The marketplace's catalogue vocabulary: what a listing can be, and the words buyers and sellers use for it in
English, French, German, Italian and Spanish.

This is product knowledge (the catalogue team maintains it). The simulated world uses the canonical values to create
listings; how people actually type them (misspellings, nicknames, other word orders) lives in the world, not here.
"""

from __future__ import annotations

LANGS = ["en", "fr", "de", "it", "es"]
MARKET_LANG = {"UK": "en", "US": "en", "FR": "fr", "DE": "de", "AT": "de", "IT": "it", "ES": "es", "NL": "en"}

CATEGORIES: dict[str, dict[str, list[str]]] = {
    "bag": {
        "en": ["bag", "handbag"],
        "fr": ["sac", "sac à main"],
        "de": ["tasche", "handtasche"],
        "it": ["borsa"],
        "es": ["bolso"],
    },
    "shoes": {
        "en": ["shoes", "pumps", "sneakers", "boots"],
        "fr": ["chaussures", "escarpins", "bottes"],
        "de": ["schuhe", "pumps", "stiefel"],
        "it": ["scarpe", "stivali"],
        "es": ["zapatos", "botas"],
    },
    "jacket": {
        "en": ["jacket", "coat", "blazer", "trench"],
        "fr": ["veste", "manteau", "trench"],
        "de": ["jacke", "mantel", "blazer"],
        "it": ["giacca", "cappotto"],
        "es": ["chaqueta", "abrigo"],
    },
    "dress": {"en": ["dress"], "fr": ["robe"], "de": ["kleid"], "it": ["vestito", "abito"], "es": ["vestido"]},
    "jewellery": {
        "en": ["bracelet", "ring", "necklace", "earrings"],
        "fr": ["bracelet", "bague", "collier"],
        "de": ["armband", "ring", "kette"],
        "it": ["bracciale", "anello"],
        "es": ["pulsera", "anillo"],
    },
    "watch": {"en": ["watch"], "fr": ["montre"], "de": ["uhr"], "it": ["orologio"], "es": ["reloj"]},
}

# brand -> canonical models (iconic lines); clothing and shoes may carry no model
BRANDS: dict[str, dict] = {
    "Chanel": {
        "aliases": ["chanel"],
        "models": {
            "Classic Flap": ["classic flap", "timeless", "2.55"],
            "Boy": ["boy bag", "boy"],
            "Wallet on Chain": ["wallet on chain", "woc"],
            "19": ["chanel 19"],
        },
    },
    "Hermès": {
        "aliases": ["hermès", "hermes"],
        "models": {"Birkin": ["birkin"], "Kelly": ["kelly"], "Constance": ["constance"], "Evelyne": ["evelyne"]},
    },
    "Louis Vuitton": {
        "aliases": ["louis vuitton", "vuitton", "lv"],
        "models": {
            "Neverfull": ["neverfull"],
            "Speedy": ["speedy"],
            "Alma": ["alma"],
            "Pochette Métis": ["pochette métis", "pochette metis", "metis"],
        },
    },
    "Dior": {
        "aliases": ["dior", "christian dior"],
        "models": {"Lady Dior": ["lady dior"], "Saddle": ["saddle"], "Book Tote": ["book tote"]},
    },
    "Gucci": {
        "aliases": ["gucci"],
        "models": {"Jackie": ["jackie"], "Marmont": ["marmont"], "Dionysus": ["dionysus"], "Horsebit": ["horsebit"]},
    },
    "Prada": {
        "aliases": ["prada"],
        "models": {"Re-Edition": ["re-edition", "re edition"], "Galleria": ["galleria"], "Cleo": ["cleo"]},
    },
    "Celine": {
        "aliases": ["celine", "céline"],
        "models": {"Luggage": ["luggage"], "Triomphe": ["triomphe"], "Belt Bag": ["belt bag"]},
    },
    "Bottega Veneta": {
        "aliases": ["bottega veneta", "bottega"],
        "models": {"Jodie": ["jodie"], "Cassette": ["cassette"], "Pouch": ["pouch"]},
    },
    "Saint Laurent": {
        "aliases": ["saint laurent", "ysl", "yves saint laurent"],
        "models": {"Loulou": ["loulou"], "Sac de Jour": ["sac de jour"], "Kate": ["kate"]},
    },
    "Fendi": {"aliases": ["fendi"], "models": {"Baguette": ["baguette"], "Peekaboo": ["peekaboo"]}},
    "Loewe": {"aliases": ["loewe"], "models": {"Puzzle": ["puzzle"], "Hammock": ["hammock"]}},
    "Balenciaga": {
        "aliases": ["balenciaga"],
        "models": {"City": ["city"], "Hourglass": ["hourglass"], "Triple S": ["triple s"]},
    },
    "Burberry": {"aliases": ["burberry"], "models": {"Trench": ["trench"]}},
    "Christian Louboutin": {
        "aliases": ["louboutin", "christian louboutin"],
        "models": {"So Kate": ["so kate"], "Pigalle": ["pigalle"]},
    },
    "Valentino": {"aliases": ["valentino"], "models": {"Rockstud": ["rockstud"]}},
    "Miu Miu": {"aliases": ["miu miu"], "models": {"Wander": ["wander"]}},
    "Cartier": {
        "aliases": ["cartier"],
        "models": {"Love": ["love bracelet", "love"], "Juste un Clou": ["juste un clou"], "Tank": ["tank"]},
    },
    "Rolex": {"aliases": ["rolex"], "models": {"Submariner": ["submariner"], "Datejust": ["datejust"]}},
    "Max Mara": {"aliases": ["max mara", "maxmara"], "models": {"Teddy": ["teddy"], "101801": ["101801"]}},
    "Zadig & Voltaire": {"aliases": ["zadig", "zadig & voltaire", "zadig et voltaire"], "models": {}},
}

# what kind of item each model is
MODEL_CATEGORY = {
    "Trench": "jacket",
    "Teddy": "jacket",
    "101801": "jacket",
    "Triple S": "shoes",
    "So Kate": "shoes",
    "Pigalle": "shoes",
    "Rockstud": "shoes",
    "Love": "jewellery",
    "Juste un Clou": "jewellery",
    "Tank": "watch",
    "Submariner": "watch",
    "Datejust": "watch",
    "Horsebit": "shoes",
}

MATERIALS: dict[str, dict[str, list[str]]] = {
    "lambskin": {"en": ["lambskin"], "fr": ["agneau"], "de": ["lammleder"], "it": ["agnello"], "es": ["cordero"]},
    "caviar": {"en": ["caviar"], "fr": ["caviar"], "de": ["caviar"], "it": ["caviar"], "es": ["caviar"]},
    "togo": {"en": ["togo"], "fr": ["togo"], "de": ["togo"], "it": ["togo"], "es": ["togo"]},
    "epsom": {"en": ["epsom"], "fr": ["epsom"], "de": ["epsom"], "it": ["epsom"], "es": ["epsom"]},
    "canvas": {
        "en": ["canvas", "monogram"],
        "fr": ["toile", "monogram"],
        "de": ["canvas", "monogram"],
        "it": ["tela", "monogram"],
        "es": ["lona", "monogram"],
    },
    "leather": {"en": ["leather"], "fr": ["cuir"], "de": ["leder"], "it": ["pelle"], "es": ["piel", "cuero"]},
    "suede": {"en": ["suede"], "fr": ["daim"], "de": ["wildleder", "veloursleder"], "it": ["camoscio"], "es": ["ante"]},
    "patent": {"en": ["patent"], "fr": ["vernis"], "de": ["lackleder", "lack"], "it": ["vernice"], "es": ["charol"]},
    "tweed": {"en": ["tweed"], "fr": ["tweed"], "de": ["tweed"], "it": ["tweed"], "es": ["tweed"]},
    "wool": {
        "en": ["wool", "cashmere"],
        "fr": ["laine", "cachemire"],
        "de": ["wolle", "kaschmir"],
        "it": ["lana", "cashmere"],
        "es": ["lana", "cachemir"],
    },
    "silk": {"en": ["silk"], "fr": ["soie"], "de": ["seide"], "it": ["seta"], "es": ["seda"]},
    "gold": {
        "en": ["gold", "yellow gold"],
        "fr": ["or", "or jaune"],
        "de": ["gold", "gelbgold"],
        "it": ["oro"],
        "es": ["oro"],
    },
    "steel": {
        "en": ["steel", "stainless steel"],
        "fr": ["acier"],
        "de": ["stahl", "edelstahl"],
        "it": ["acciaio"],
        "es": ["acero"],
    },
}

COLOURS: dict[str, dict[str, list[str]]] = {
    "black": {
        "en": ["black"],
        "fr": ["noir", "noire"],
        "de": ["schwarz", "schwarze"],
        "it": ["nero", "nera"],
        "es": ["negro", "negra"],
    },
    "white": {
        "en": ["white", "ecru"],
        "fr": ["blanc", "blanche", "écru"],
        "de": ["weiß", "weiss"],
        "it": ["bianco", "bianca"],
        "es": ["blanco", "blanca"],
    },
    "beige": {
        "en": ["beige", "camel"],
        "fr": ["beige", "camel"],
        "de": ["beige", "camel"],
        "it": ["beige"],
        "es": ["beige", "camel"],
    },
    "brown": {
        "en": ["brown", "tan", "cognac"],
        "fr": ["marron", "cognac"],
        "de": ["braun", "cognac"],
        "it": ["marrone", "cuoio"],
        "es": ["marrón", "cuero"],
    },
    "red": {
        "en": ["red", "burgundy"],
        "fr": ["rouge", "bordeaux"],
        "de": ["rot", "bordeaux"],
        "it": ["rosso", "rossa"],
        "es": ["rojo", "roja"],
    },
    "pink": {"en": ["pink"], "fr": ["rose"], "de": ["rosa", "pink"], "it": ["rosa"], "es": ["rosa"]},
    "blue": {
        "en": ["blue", "navy"],
        "fr": ["bleu", "bleue", "marine"],
        "de": ["blau", "navy"],
        "it": ["blu", "azzurro"],
        "es": ["azul"],
    },
    "green": {"en": ["green"], "fr": ["vert", "verte"], "de": ["grün", "gruen"], "it": ["verde"], "es": ["verde"]},
    "gold_colour": {
        "en": ["gold tone", "golden"],
        "fr": ["doré", "dorée"],
        "de": ["golden"],
        "it": ["dorato"],
        "es": ["dorado"],
    },
    "grey": {"en": ["grey", "gray"], "fr": ["gris"], "de": ["grau"], "it": ["grigio"], "es": ["gris"]},
}

STYLES: dict[str, dict[str, list[str]]] = {  # soft attributes a buyer may ask for
    "quilted": {
        "en": ["quilted"],
        "fr": ["matelassé", "matelasse"],
        "de": ["gesteppt", "gesteppte"],
        "it": ["trapuntata", "matelassé"],
        "es": ["acolchado", "acolchada"],
    },
    "mini": {
        "en": ["mini", "small"],
        "fr": ["mini", "petit"],
        "de": ["mini", "klein"],
        "it": ["mini", "piccola"],
        "es": ["mini", "pequeño"],
    },
    "vintage": {"en": ["vintage"], "fr": ["vintage"], "de": ["vintage"], "it": ["vintage"], "es": ["vintage"]},
}

CONDITIONS = ["never worn", "very good", "good", "fair"]
SHOE_SIZES = [str(s) for s in range(35, 43)]
CLOTHING_SIZES = ["XS", "S", "M", "L", "XL"]
BAG_SIZES = ["25", "30", "35"]  # Birkin and Kelly sizes in cm; other bags carry none


def surface_forms(table: dict[str, dict[str, list[str]]]) -> dict[str, str]:
    """Every listed word in every language -> its canonical value."""
    return {w: k for k, langs in table.items() for ws in langs.values() for w in ws}


def brand_forms() -> dict[str, str]:
    return {a: b for b, v in BRANDS.items() for a in v["aliases"]}


def model_forms() -> dict[str, tuple[str, str]]:
    """model words -> (brand, model)."""
    return {w: (b, m) for b, v in BRANDS.items() for m, ws in v["models"].items() for w in ws}


def category_of_model(model: str) -> str:
    return MODEL_CATEGORY.get(model, "bag")
