# Flappy Bird

![](https://github.com/AmpiroMax/FlappyBirdSOLO/blob/master/assets/2023-05-23%2015-17-21.gif)

## Что хотелось сделать

Запускаешь мега крутого бота -> он играет лучше человека

## Что получилось

Есть модель, котороя берет информацию о координатах игрока и труб и, играя на упрощенной версии игры, может достигать результатов в 300 очков. Может и больше может, но дальше уже останавливал. Это лучший резлультат, который получилось достичь.

Есть модель, работающая напрямую с изображением игры. Она не смогла обучиться нормально проходить игру.

## Что использовалось

Решение задачи было принципиально получить с помощью DQN. Поэтому в данной работе присутствует только этот подход. Для улучшения результатов были применены различные лайфхаки:

- Replay buffer
- Double DQN
- E-greedy policy

Также по итогу было решено попробвать решать задачу через CV. На вход подается несколько изображений, чтобы передать временное свойство. Таким образом, можно добавить ещё одну фичу:

- Temporal difference

Модель, принимавшая на вход изображения не смогла достич весомых результатов. Говоря простым языком, он тупила как черт.
