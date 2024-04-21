## Surname Simulation

苗字絶滅問題のシミュレーション用プログラムです. 作りかけです.

### インストール方法

Windowsの人はWSLとかで動かしてください. c++コンパイラ, cmake, GSLなどが必要です.

```sh
sudo apt install build-essential
sudo apt install libgsl-dev
sudo apt install cmake
pip install git+https://github.com/Gedevan-Aleksizde/surname-sim
```

苗字別人口データは以下のようにして取得できます. pandas.DataFrame 型です.

```python
from surname_sim.loader import load_dataset

d = load_dataset()
```

苗字別人口の出典は『苗字由来net』の[全国名字ランキング](https://myoji-yurai.net/prefectureRanking.htm)の1位から40000位までです. 2024年4月中旬ごろ閲覧しました. 利用の際は出典となったサイトの利用規約も確認してください.

### 使用例


```python
from surname_sim.sim import SimpleSimulator

gen = SimpleSimulator(seed=42)
gen.nextRandomMN(d['number'].values)

gen.iterateChunkHG([10, 20, 30], 10)
```