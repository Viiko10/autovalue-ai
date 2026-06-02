# AutoValue AI

**Semesterarbeit — ZHAW Wirtschaftsinformatik, Modul KI-Anwendungen, FS2026**

AutoValue AI schätzt den Marktpreis eines Gebrauchtwagens anhand eines Fotos und strukturierter Fahrzeugdaten (Marke, Modell, Baujahr, Kilometerstand, Kraftstoff, Getriebe) und erklärt die Schätzung in natürlicher Sprache.

Das Projekt kombiniert drei KI-Blöcke zu einer durchgehenden Pipeline: Ein Computer-Vision-Modell (CLIP) bewertet den Fahrzeugzustand aus dem Foto und gibt einen numerischen Zustandsscore zurück. Dieser Score fliesst zusammen mit den Fahrzeugdaten in ein trainiertes GradientBoosting-Modell ein, das den Preis schätzt. Abschliessend erklärt GPT-4o-mini die Schätzung auf Basis ähnlicher Fahrzeuge aus dem Trainingsdatensatz (RAG) und steht für Folgefragen als Chatbot zur Verfügung.

Das Modell wurde auf dem UK Used Cars Dataset von Kaggle trainiert (~100'000 Einträge, 11 Fahrzeugmarken). Die vollständige Dokumentation befindet sich in [DOCUMENTATION.md](DOCUMENTATION.md), die laufende Demo auf [HuggingFace Spaces](https://huggingface.co/spaces/Viiko10/autovalue-ai).
