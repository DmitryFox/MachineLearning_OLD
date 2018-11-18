
# Непараметрическая регрессия.<br />Формула Надарая-Ватсона. Метод LOWESS.

## Функция Ядра:
В [непараметрической статистике](https://ru.wikipedia.org/wiki/%D0%9D%D0%B5%D0%BF%D0%B0%D1%80%D0%B0%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D1%81%D1%82%D0%B0%D1%82%D0%B8%D1%81%D1%82%D0%B8%D0%BA%D0%B0 "Непараметрическая статистика") под ядром понимается весовая функция, используемая при оценке распределений и параметров ([ядерная оценка плотности](https://ru.wikipedia.org/wiki/%D0%AF%D0%B4%D0%B5%D1%80%D0%BD%D0%B0%D1%8F_%D0%BE%D1%86%D0%B5%D0%BD%D0%BA%D0%B0_%D0%BF%D0%BB%D0%BE%D1%82%D0%BD%D0%BE%D1%81%D1%82%D0%B8 "Ядерная оценка плотности"), [ядерная регрессия](https://ru.wikipedia.org/wiki/%D0%AF%D0%B4%D0%B5%D1%80%D0%BD%D0%B0%D1%8F_%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F "Ядерная регрессия")).  Ядерная оценка плотности является задачей сглаживания данных. Смысл ядерной регрессии заключается в поиске нелинейного отношения между парой случайных величин **X** и **Y**. Ядерная оценка требует специфицировать ширину окна.

**В задаче непараметрической регрессии были реализованы следующие ядра на языке Python:**
* Гауссовское ядро:<br />
![equation](https://latex.codecogs.com/gif.latex?{\displaystyle%20K(u)={\frac%20{1}{\sqrt%20{2\pi%20}}}e^{-{\frac%20{1}{2}}u^{2}}})
* Квартическое ядро (оно же биквадратичное):<br />
![equation](https://latex.codecogs.com/gif.latex?{\displaystyle%20K(u)={\frac%20{15}{16}}(1-u^{2})^{2}})<br />
Носитель: ![equation](https://latex.codecogs.com/gif.latex?{\displaystyle%20|u|\leq%201})

## Формула Надарая-Ватсона:

Формула Надарая-Ватсона используется для решения задачи непараметрического [восстановления регрессии.](http://www.machinelearning.ru/wiki/index.php?title=%D0%A0%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D0%BE%D0%BD%D0%BD%D1%8B%D0%B9_%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7 "Восстановление регрессии")

Реализована формула Надарая-Ватсона на языке python:<br />
![equation](https://latex.codecogs.com/gif.latex?a_h(x;X^l)%20=%20\frac{\sum_{i=1}^{l}%20y_i\omega_i(x)}{\sum_{i=1}^{l}%20\omega_i(x)}%20=%20\frac{\sum_{i=1}^{l}%20y_iK\left(\frac{\rho(x,x_i)}{h}%20\right%20)}{\sum_{i=1}^{l}%20K\left(\frac{\rho(x,x_i)}{h}%20\right%20)})

Функция находиться в файле `regression.py`
```python
def nadaraya_watson(value, x, y, h, kernel, metric):
```
Принимает аргументы: `value` - искомое, оптимальное значение в точке x;  `x` - вектор объектов;  `y` - вектор ответов; `h` - коэффициент сглаживания (ширина окна); `kernel` - функция ядра; `metric` - функция метрики (находящая длину).

## Метод LOWESS

Оценка Надарайя–Ватсона крайне чувствительна к большим одиночным выбросам. Идея обнаружения выбросов заключается в том, что чем больше величина ошибки, тем в большей степени прецеддент является выбросом, и тем меньше должен быть его вес.

Алгоритм LOWESS  выглядит следующим образом:
* Вход:
	* ![equation](https://latex.codecogs.com/gif.latex?X^m) — обучающая выборка;
	* ![equation](https://latex.codecogs.com/gif.latex?w_t,%20\,\,\,%20t=1,\ldots,m) весовые функции;
* Выход:
	* коэффициенты ![equation](https://latex.codecogs.com/gif.latex?\delta_t,%20\,\,\,%20t=1,\ldots,m)
#
1. Инициализировать ![equation](https://latex.codecogs.com/gif.latex?\gamma_1:=\ldots=\gamma_m:=1)
2. **повторять**
3. Вычислить оценки скользящего контроля на каждом объекте: <br />
![equation](https://latex.codecogs.com/gif.latex?a_i:=a_h(x_i;%20X^\ell\setminus\{x_i\})%20=%20\frac{%20\sum_{i=1,%20i\neq%20t%20}^{m}%20{y_i%20\gamma_i%20K\left(%20\frac{\rho(x_i,x_t)}%20{h(x_t)}\right)}}%20{\sum_{i=1,%20i\neq%20t%20}^{m}%20{\gamma_i%20K\left(%20\frac{\rho(x_i,x_t)}{h(x_t)}\right)}%20})
4. Вычислить коэфициенты: ![equation](https://latex.codecogs.com/gif.latex?\gamma_i)<br />
![equation](https://latex.codecogs.com/gif.latex?\gamma_i:=%20\tilde{K}(|%20a_i%20-%20y_i%20\|);%20i%20=%201,...,\ell;)
5. **пока** коэффициенты ![equation](https://latex.codecogs.com/gif.latex?\gamma_i) не стабилизируются;

<br />

> Использованная литература:
> 1) MachineLearning.ru - http://www.machinelearning.ru/wiki/index.php?title=%D0%A4%D0%BE%D1%80%D0%BC%D1%83%D0%BB%D0%B0_%D0%9D%D0%B0%D0%B4%D0%B0%D1%80%D0%B0%D1%8F-%D0%92%D0%B0%D1%82%D1%81%D0%BE%D0%BD%D0%B0)
> 2) Википедия - https://ru.wikipedia.org/wiki/%D0%AF%D0%B4%D1%80%D0%BE_(%D1%81%D1%82%D0%B0%D1%82%D0%B8%D1%81%D1%82%D0%B8%D0%BA%D0%B0)
> 3) Математические методы обучения по прецедентам (теория обучения машин) К. В. Воронцов

