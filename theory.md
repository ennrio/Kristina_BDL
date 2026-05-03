# Теоретическая часть — ЛР 2
**Студент:** Пресняков Егор Степанович  
**Student ID:** 2  
**Вариант:** датасет airfoil_self_noise

---

## Вопрос 1. Градиент функционала качества

Рассмотрим линейную модель и функционал качества MSE:

$$
Q(\theta, X^{\ell}) = \frac{1}{\ell} \sum_{i=1}^{\ell} \left( \sum_{j=1}^{d} \theta_j x_{i,j} + \theta_0 - y_i \right)^2
$$

где $\theta = (\theta_1, \dots, \theta_d)$ — вектор параметров, $\theta_0$ — смещение (bias).

---

### 1. Вывод $\dfrac{\partial Q}{\partial \theta_j}$ и матричная форма

#### Пошаговый вывод

1. **Линейность производной**  
   $$
   \frac{\partial Q}{\partial \theta_j} = \frac{1}{\ell} \sum_{i=1}^{\ell} \frac{\partial}{\partial \theta_j} \left( \sum_{k=1}^{d} \theta_k x_{i,k} + \theta_0 - y_i \right)^2
   $$

2. **Цепное правило**  
   $$
   = \frac{1}{\ell} \sum_{i=1}^{\ell} 2 \left( \sum_{k=1}^{d} \theta_k x_{i,k} + \theta_0 - y_i \right) \cdot \frac{\partial}{\partial \theta_j} \left( \sum_{k=1}^{d} \theta_k x_{i,k} \right)
   $$

3. **Производная линейной формы**  
   $\frac{\partial}{\partial \theta_j} \sum_{k=1}^{d} \theta_k x_{i,k} = x_{i,j}$, а $\theta_0$ и $y_i$ не зависят от $\theta_j$.

4. **Итоговая скалярная формула**  
   $$
   \frac{\partial Q}{\partial \theta_j} = \frac{2}{\ell} \sum_{i=1}^{\ell} \left( \sum_{k=1}^{d} \theta_k x_{i,k} + \theta_0 - y_i \right) x_{i,j}
   $$

#### Матричная форма

- $X \in \mathbb{R}^{\ell \times d}$ — матрица признаков  
- $\mathbf{y} \in \mathbb{R}^{\ell}$ — вектор ответов  
- $\mathbf{1}$ — вектор из единиц  
- $\hat{\mathbf{y}} = X\theta + \theta_0\mathbf{1}$  
- $\mathbf{r} = \hat{\mathbf{y}} - \mathbf{y}$

$$
\nabla_{\theta} Q = \frac{2}{\ell} X^\top \mathbf{r} = \frac{2}{\ell} X^\top \left( X\theta + \theta_0 \mathbf{1} - \mathbf{y} \right)
$$

---

### 2. Эквивалентность записей и градиент по $\theta_0$

Если добавить константный признак $x_{i,0}=1$ и расширить вектор параметров до $\tilde{\theta} = (\theta_0, \theta_1, \dots, \theta_d)^\top$, то скалярное произведение $\langle \tilde{\theta}, \tilde{\mathbf{x}}_i \rangle$ в точности равно $\theta_0 + \sum_{j=1}^d \theta_j x_{i,j}$.

Градиент по $\theta_0$ получается как частный случай общей формулы при $j=0$ и $x_{i,0}=1$:

$$
\frac{\partial Q}{\partial \theta_0} = \frac{2}{\ell} \sum_{i=1}^{\ell} \left( \sum_{j=1}^{d} \theta_j x_{i,j} + \theta_0 - y_i \right) \cdot 1
$$

---

### 3. SGD vs Полный градиентный спуск

| Аспект | Полный GD | SGD |
|--------|-----------|-----|
| Точность градиента | точный | несмещённый, высокая дисперсия |
| Сложность шага | $O(\ell d)$ | $O(d)$ |
| Траектория | гладкая | шумная |
| Сходимость | медленно на больших данных | быстро |

> **Итог**: на практике предпочитают SGD (или Adam) из-за скорости и масштабируемости.

---

## Вопрос 2. Регуляризация L1 и L2

Рассмотрим регуляризованный функционал ($\tau > 0$):

$$
Q_\tau(\theta, X^{\ell}) = Q(\theta, X^{\ell}) + \tau \cdot R(\theta)
$$

(смещение $\theta_0$ обычно не регуляризуют)

---

### 1. Формулы регуляризаторов и добавка в градиент

#### Ridge (L2)

- $R_{\text{L2}}(\theta) = \|\theta\|_2^2 = \sum_{j=1}^d \theta_j^2$
- Добавка в градиент: $\frac{\partial}{\partial \theta_j} \left( \tau \sum_{k=1}^d \theta_k^2 \right) = 2\tau \theta_j$
- Обновление SGD: $\theta_j \leftarrow \theta_j - \eta \left( \frac{\partial Q}{\partial \theta_j} + 2\tau \theta_j \right)$

#### LASSO (L1)

- $R_{\text{L1}}(\theta) = \|\theta\|_1 = \sum_{j=1}^d |\theta_j|$
- Добавка в градиент (при $\theta_j \neq 0$): $\tau \cdot \text{sign}(\theta_j)$
- Обновление SGD: $\theta_j \leftarrow \theta_j - \eta \left( \frac{\partial Q}{\partial \theta_j} + \tau \cdot \text{sign}(\theta_j) \right)$

---

### 2. Геометрическая интерпретация

Эквивалентная задача: $\min_{\theta} Q(\theta)$ при $R(\theta) \le t$.

- **L2**: область — шар → касание в точке с ненулевыми координатами.
- **L1**: область — ромб (многогранник) → касание часто происходит в углу → нулевые координаты (разреженность).

---

### 3. Поведение при $\tau \to 0$ и $\tau \to +\infty$

| Предел | Ridge | LASSO |
|--------|-------|-------|
| $\tau \to 0$ | $\theta \to \theta_{\text{OLS}}$ | $\theta \to \theta_{\text{OLS}}$ |
| $\tau \to +\infty$ | $\theta \to 0$ асимптотически | $\theta \to 0$ точно, с обнулением признаков |

---

### 4. Проблема sign(0) и её решение

Функция $|x|$ не дифференцируема в нуле. На практике используют:

1. **Soft‑thresholding** (проксимальный оператор):  
   $$
   \theta_j \leftarrow \mathcal{S}_{\eta\tau}\left( \theta_j - \eta \frac{\partial Q}{\partial \theta_j} \right), \quad \mathcal{S}_\lambda(z) = \text{sign}(z) \cdot \max(|z| - \lambda, 0)
   $$
2. Субградиентный метод с $\text{sign}(0)=0$ (медленнее).  
3. Координатный спуск.  
4. Сглаживание $|x| \approx \sqrt{x^2+\varepsilon}$.

---

## Вопрос 3. Анализ результатов на своём датасете

### 1. Влияние стандартизации

Признаки `frequency` (сотни Гц), `velocity` (десятки м/с), `thickness` (сотые доли метра) имеют разные масштабы. Без стандартизации матрица $X^\top X$ плохо обусловлена ($\kappa \gg 1$). Изолинии вытянуты, SGD "метается". После стандартизации $\text{Var}(x_j) \approx 1$, $\kappa \approx 1$, изолинии становятся окружностями — сходимость ускоряется в десятки раз.

### 2. Экспериментальное подтверждение

- **Без стандартизации**: при $\eta=0.01$ — overflow, веса в `inf`. При $\eta=10^{-4}$ loss колеблется, MAPE > 0.15.
- **Со стандартизацией**: $\eta=0.01$, batch=32 — монотонное падение MSE, MAPE ≈ 0.029 (compare с sklearn 0.0298).

**Вывод:** стандартизация необходима для градиентных методов на разномасштабных данных.