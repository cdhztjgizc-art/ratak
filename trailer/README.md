# Trailer – ČERNÝ OREL

Procedurálně generovaný trailer (obraz, hudba i zvukové efekty) – nepotřebuje žádné externí assety.

## Spuštění

```bash
pip install -r requirements.txt   # + ffmpeg v systému
python3 make_trailer.py            # -> output/cerny_orel_trailer.mp4 (66 s, 1080p, 30 fps)
```

Další možnosti:

- `python3 make_trailer.py --stills 12 25.5` – náhledové PNG snímky v zadaných časech (s)
- `python3 make_trailer.py --audio-only` – jen zvuková stopa

## Úpravy

Všechny texty jsou v `config.json`:

| Klíč | Co mění |
| --- | --- |
| `title`, `tagline`, `release` | název hry, slogan, „JIŽ BRZY“ |
| `intro_lines` | úvodní věty |
| `gameplay_captions` | titulky ke 3 gameplay záběrům |
| `modes` | herní módy (`icon`: `crosshair`, `flag`, `bomb`, `skull`) |
| `weapons` | zbraně ve zbrojírně (`kind`: `ar`, `sniper`, `smg`; statistiky 0–1) |
| `attachments` | attachmenty (`slot`: `optic`, `muzzle`, `grip`, `mag`, `laser`) |

## Struktura (sekundy)

| Čas | Scéna |
| --- | --- |
| 0–6 | intro |
| 6–10 | logo |
| 10–22 | gameplay (útočná puška, odstřelovačka, výbuch) |
| 22–34 | herní módy |
| 34–45 | zbrojírna |
| 45–57 | attachmenty |
| 57–59 | rychlý střih |
| 59–66 | finále |
