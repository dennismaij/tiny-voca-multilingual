#!/usr/bin/env python3
"""Generate a high-density Dutch Technical & Borrowed Words dataset.

Creates diverse sentences containing:
- Tech loanwords (edge cases, happy path, drag, prompt, pipeline, dashboard, etc.)
- Latinate & complex verbs (-eren, -erend, opererend, analyserend, etc.)
- Infosec & systems analysis (stresstests, risicoanalyse, faalmodi, etc.)
- Modern business & product engineering terms
"""

import random
from pathlib import Path

SUBJECTS = [
    "Het engineeringteam", "Onze lead architect", "Het platform", "De security analyst",
    "Onze data pipeline", "De backend service", "Het monitoring systeem", "Onze infrastructuur",
    "De senior developer", "Het DevOps team", "De auditcommissie", "Onze AI-assistent",
    "Het detectiesysteem", "De product owner", "Het QA-team", "De forensic investigator"
]

ACTIONS = [
    "identificeert dagelijks verborgen risico's en",
    "voert diepgaande stresstests uit op",
    "analyseert alle kritieke edge cases binnen",
    "minimaliseert onnodige systemische drag in",
    "voorkomt fatale faalmodi tijdens",
    "optimaliseert de runtime performance van",
    "is continu opererend om",
    "beoordeelt complexe architecturen voor",
    "signaleert blinde vlekken en kwetsbaarheden in",
    "automatiseert het complete releaseproces voor",
    "verifieert alle security controls van",
    "brengt structurele wrijving in kaart bij"
]

OBJECTS = [
    "de microservices architectuur",
    "de hybride cloudomgeving",
    "onze gedistribueerde datastores",
    "het authenticatie- en autorisatieframework",
    "de real-time event streaming pipeline",
    "alle API gateways en ingress controllers",
    "het geautomatiseerde deployment script",
    "onze Kubernetes clusters in productie",
    "de performante caching laag",
    "het forensische logging mechanisme",
    "de zero-trust security perimeter",
    "onze low-latency inference endpoints"
]

CONCLUSIONS = [
    "om de stabiliteit van het systeem te waarborgen.",
    "waardoor onverwachte crashes worden geëlimineerd.",
    "zodat de happy path naadloos blijft functioneren.",
    "voordat updates naar productie worden uitgerold.",
    "waarmee dataverlies effectief wordt voorkomen.",
    "en de algehele gebruikerservaring verbetert.",
    "terwijl de operationele overhead afneemt.",
    "zonder dat dit leidt tot onnodige frictie.",
    "met een heldere focus op meetbare resultaten.",
    "omvat een grondige risicoanalyse en foutanalyse."
]

VOCAB_SENTENCES = [
    "Ik ben Umbris, opererend door de Occlusion Lens.",
    "Mijn primaire doel is forensische stresstests, risicoanalyse en protectieve foutanalyse.",
    "We identificeren verborgen gevaren, systemische drag, blinde vlekken, edge cases en fatale faalmodi.",
    "Het artikel vermijdt clichés en kiest voor een forensisch handboek over de kosten van ruis.",
    "Elk overbodig woord is een lek in de aandacht van de lezer.",
    "Dat is precies het happy path dat een standaard model produceert.",
    "Wanneer een concept te complex is, ontstaat er onnodige systemische wrijving.",
    "De enige weg naar helderheid is een meedogenloze snoei van alle ballast.",
    "We testen alle edge cases voordat we de nieuwe release live zetten.",
    "De container service draait stabiel in onze Kubernetes cluster.",
    "Goede prompt engineering voorkomt hallucinaties en verhoogt de kwaliteit.",
    "Onze CI/CD pipeline bouwt en deployt microservices binnen enkele minuten.",
    "Het dashboard geeft direct inzicht in CPU-verbruik, latency en memory leaks.",
    "Tijdens de security audit vonden we geen enkele kritieke exploit of zero-day.",
    "Zinsser benadrukte dat schrijven neerkomt op het weghalen van overtollige ruis.",
    "De data engineers optimaliseren de ETL pipeline voor BigQuery en Snowflake.",
    "Het systeem is continu opererend en voert realtime anomaly detection uit.",
    "Complexe foutanalyse vereist een gestructureerde aanpak van het probleem.",
    "De pull request bevat unit tests voor alle zeldzame faalmodi.",
    "Door de serverless architectuur schaalt de backend automatisch op bij piekmomenten."
]


def generate_corpus(count=1200, output_path="datasets/dutch_tech_studio/corpus.txt"):
    random.seed(42)
    sentences = list(VOCAB_SENTENCES)
    
    while len(sentences) < count:
        s = f"{random.choice(SUBJECTS)} {random.choice(ACTIONS)} {random.choice(OBJECTS)} {random.choice(CONCLUSIONS)}"
        if s not in sentences:
            sentences.append(s)
            
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for sent in sentences[:count]:
            f.write(sent.strip() + "\n")
            
    print(f"Generated {len(sentences[:count])} Dutch technical sentences at {out}")


if __name__ == "__main__":
    generate_corpus()
