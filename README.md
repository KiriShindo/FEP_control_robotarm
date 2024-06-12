# FEP_control_robotarm

## 概要
自由エネルギー原理に基づいてロボットアームを制御することで，ロボットアームの身体構造が変化したときに適応的な動作が生成されるかを確かめた．

## 自由エネルギー原理
自由エネルギー原理とはFristonによって提唱された脳の情報処理に関する統一理論であり，知覚，行動，学習が変分自由エネルギーの最小化によって実現されるとしている．
自由エネルギー原理に従う生物は，外界の物理法則を脳内で模倣し，外界の事象を再現している．
そして，外界から得られる感覚情報と再現される感覚情報の誤差が小さくなるように，自己の認識の更新や行動選択を行っている．  
![スクリーンショット 2024-06-06 151637](https://github.com/KiriShindo/FEP_control_robotarm/assets/170800970/655929f5-197d-452c-acef-dee858845291)


## 構成
### fig フォルダ



- 各種ロボットアームに関する実験結果
- 制御モデルの図

### src フォルダ
- 各種ロボットアームに関するシミュレーションコード

## シミュレーション環境
- **シミュレーションソフト**: MuJoCo
- **OS**: Windows 11
- **GPU**: GeForce RTX™ 4060 Ti 16GB

## シミュレーションの様子
###通常状態のシミュレーション
*＜アームのシミュレーション＞*
![movie_with_image](https://github.com/KiriShindo/FEP_control_robotarm/assets/170800970/9abaa510-e897-49c8-a372-6146843dafc2)

*＜誤差平面上の軌跡：青い場所ほどMSEが大きく，赤い場所ほどMSEが小さい＞*
![error_field](https://github.com/KiriShindo/FEP_control_robotarm/assets/170800970/cbe6ce5d-d285-429d-92a7-cbb6d4a3edd7)

*＜2つを合わせた動画＞*
![combined](https://github.com/KiriShindo/FEP_control_robotarm/assets/170800970/46cf7b2f-9c8d-46a7-aeff-ae7583b0224b)

###アーム長が変化した後のシミュレーション
*＜アームのシミュレーション＞*
![movie_with_image_1](https://github.com/KiriShindo/FEP_control_robotarm/assets/170800970/73c1c0f2-380b-429c-bbfb-3c4c978abb1c)

*＜誤差平面上の軌跡：青い場所ほどMSEが大きく，赤い場所ほどMSEが小さい＞*
![error_field_p2_1](https://github.com/KiriShindo/FEP_control_robotarm/assets/170800970/86dd8f5b-7658-4b82-a8f6-23e6a26f82e8)

*＜2つを合わせた動画＞*
![combined (1)](https://github.com/KiriShindo/FEP_control_robotarm/assets/170800970/72361f9a-cbd8-4bad-b99a-11f70a46d4a3)

