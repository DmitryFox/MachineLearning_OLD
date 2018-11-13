
# Задача восстановления регрессии.
## Непараметрическая регрессия.<br />Формула Надарая-Ватсона. Метод LOWESS.

**Формула Надарая-Ватсона:**

Формула Надарая-Ватсона используется для решения задачи непараметрического [восстановления регрессии.](http://www.machinelearning.ru/wiki/index.php?title=%D0%A0%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D0%BE%D0%BD%D0%BD%D1%8B%D0%B9_%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7 "Восстановление регрессии")

Реализована формула Надарая-Ватсона на языке python:
![enter image description here](http://www.machinelearning.ru/mimetex/?$a_h%28x;X%5El%29%20=%20%5Cfrac%7B%5Csum_%7Bi=1%7D%5E%7Bl%7D%20y_i%5Comega_i%28x%29%7D%7B%5Csum_%7Bi=1%7D%5E%7Bl%7D%20%5Comega_i%28x%29%7D%20=%20%5Cfrac%7B%5Csum_%7Bi=1%7D%5E%7Bl%7D%20y_iK%5Cleft%28%5Cfrac%7B%5Crho%28x,x_i%29%7D%7Bh%7D%20%5Cright%20%29%7D%7B%5Csum_%7Bi=1%7D%5E%7Bl%7D%20K%5Cleft%28%5Cfrac%7B%5Crho%28x,x_i%29%7D%7Bh%7D%20%5Cright%20%29%7D$)

Функция находиться в файле `regression.py`
```python
def nadaraya_watson(value, x, y, h, kernel, metric):  
```
Принимает аргументы:
* `value` - значение x из выборки X
* `x` - выборка X
* `y` - выборка Y
* `h` - коэффициент сглаживания
* `kernel` - функция ядра
* `metric` - функция метрики


> Список литературы:
> 1) MachineLearning.ru - http://www.machinelearning.ru/wiki/index.php?title=%D0%A4%D0%BE%D1%80%D0%BC%D1%83%D0%BB%D0%B0_%D0%9D%D0%B0%D0%B4%D0%B0%D1%80%D0%B0%D1%8F-%D0%92%D0%B0%D1%82%D1%81%D0%BE%D0%BD%D0%B0)
> 2) Математические методы обучения по прецедентам (теория обучения машин) К. В. Воронцов

**Метод LOWESS**

Алгоритм LOWESS  выглядит следующим образом:
1. Инициализировать ![enter image description here](http://www.machinelearning.ru/mimetex/?\delta_1:=\ldots=\delta_m:=1)

**2. повторять**

3. Вычислить оценки скользящего контроля на каждом объекте 

![enter image description here](http://www.machinelearning.ru/mimetex/?%20\hat{y_t}:=a(x_t;%20X\setminus\{%20x_t\})%20=%20\frac{%20\sum_{i=1,%20i\neq%20t%20}^{m}%20{y_i%20\delta_i%20K\left(%20\frac{\rho(x_i,x_t)}{h(x_t)}\right)}%20}%20{\sum_{i=1,%20i\neq%20t%20}^{m}%20{y_i%20K\left(%20\frac{\rho(x_i,x_t)}{h(x_t)}\right)}%20})
  
4. По набору значений \hat{\varepsilon_t}= \| \hat{y_t} - y_t \| вычислить новые значения коэффициентов \delta_t. 

5. пока веса \delta_t не стабилизируются 

> 1) MachineLearning.ru - http://www.machinelearning.ru/wiki/index.php?title=%D0%90%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_LOWESS


**Функция Ядра:**
В [непараметрической статистике](https://ru.wikipedia.org/wiki/%D0%9D%D0%B5%D0%BF%D0%B0%D1%80%D0%B0%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D1%81%D1%82%D0%B0%D1%82%D0%B8%D1%81%D1%82%D0%B8%D0%BA%D0%B0 "Непараметрическая статистика") под ядром понимается весовая функция, используемая при оценке распределений и параметров ([ядерная оценка плотности](https://ru.wikipedia.org/wiki/%D0%AF%D0%B4%D0%B5%D1%80%D0%BD%D0%B0%D1%8F_%D0%BE%D1%86%D0%B5%D0%BD%D0%BA%D0%B0_%D0%BF%D0%BB%D0%BE%D1%82%D0%BD%D0%BE%D1%81%D1%82%D0%B8 "Ядерная оценка плотности"), [ядерная регрессия](https://ru.wikipedia.org/wiki/%D0%AF%D0%B4%D0%B5%D1%80%D0%BD%D0%B0%D1%8F_%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F "Ядерная регрессия")).  Ядерная оценка плотности является задачей сглаживания данных. Смысл ядерной регрессии заключается в поиске нелинейного отношения между парой случайных величин **X** и **Y**. Ядерная оценка требует специфицировать ширину окна.

**В задаче непараметрической регрессии были реализованы следующие ядра на языке Python:**
* Гауссовское ядро:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/2823201b13dc500e1a48d7907c3c64c2ad82395d)

* Квартическое ядро (оно же биквадратичное):

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/b0c8b60cf84bc6bcdaa124b727e382b0716c033d)

Носитель:![](https://wikimedia.org/api/rest_v1/media/math/render/svg/63aeef556c12f0f18861dbeee4d1342237989334)

> Список литературы:
> 1) Википедия - https://ru.wikipedia.org/wiki/%D0%AF%D0%B4%D1%80%D0%BE_(%D1%81%D1%82%D0%B0%D1%82%D0%B8%D1%81%D1%82%D0%B8%D0%BA%D0%B0)

