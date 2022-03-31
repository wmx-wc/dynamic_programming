# dynamic_programming
# 动态规划算法
## 用途和步骤
### 用途
动态规划通过组合子问题的解来求解原问题。

分治方法将问题划分为互不相交的子问题，递归地求解子问题，然后将它们的解组合起来，求出原问题的解。

动态规划应用于子问题重叠的情况，即不同的子问题具有公共的子子问题，（子问题的求解是递归进行的，将其划分为更小的子子问题）。
在这种情况下，分治算法会做许多不必要的工作，它会反复地求解那些公共子问题，而动态规划算法只会对子子问题求解一次，无需每次求解子子问题都重新计算。

动态规划算法通常用来求解最优化问题。这类问题有很多可行的解，每个解都有一个值，找出最优值的解。

### 步骤
设计动态规划算法的四个步骤：

1、刻画一个最优解的结构特征 

2、递归地定义最优解的值

3、计算最优解的值，通常采用自底向上的方法

4、利用计算出的信息构造一个最优解

如果我们只需要求解一个最优解的值而非解本身，可以忽略步骤四。如果要求解所有的最优解，就要做步骤四，有时就需要在执行步骤三的过程中维护一些额外的信息，以便用来构造一个最优解。

## 重叠子问题
**最优子结构：问题的最优解由相关子问题的最优解组合而成，而这些子问题可以独立求解。**

## 钢条切割

```java
public int cut_rod(int[] p, int n) {
    if (n == 0) {
        return 0;
    }
    int q = Integer.MIN_VALUE;
    for (int i = 0; i < Math.min(n, p.length); i++) {
        // 最优子结构
        q = Math.max(q, p[i] + cut_rod(p, n - i - 1));
    }
    return q;
}
```
自顶向下的递归方法。`cut_rod`函数反复使用相同的参数值对自身进行递归调用，即它反复求解相同的子问题。

朴素递归方法之所以效率很低，是因为它反复求解相同的子问题。动态规划方法仔细安排求解顺序，对每个子问题只求解一次，并将结果保存下来，如果随后再次需要此问题的解，只需要查找保存的结果，不必重新计算。

因此，动态规划方法是付出额外的内存空间来节省计算时间，是典型的时空权衡的例子。

如果子问题的的数量是输入规模的多项式函数，而我们可以在多项式时间内求解出每个子问题，那么动态规划算法的总时间是多项式阶的。

## 动态规划实现方法
### 带备忘的自顶向下法
此方法仍然按照自然的递归形式编写过程，但过程会保存每个子问题的解（通常保存在一个数组或者散列表中）。当需要一个子问题的解时，过程首先检查是否已经保存过此解，如果是，则直接返回保存的值，从而节省了计算时间；否则，按通常方式解决子问题。

### 自底向上法

此方法一般需要恰当定义子问题“规模”的概念，使得任何子问题的求解都只依赖于“更小的”子问题的求解。我们可以将子问题按照规模进行排序，按由小到大的顺序进行求解。当求解某个子问题时，它所依赖的那些更小的子问题都已经求解完毕，结果已经保存。每个子问题只需要求解一次，当我们求解它时（第一次遇到），它所依赖的那些前提子问题都已经求解完成。

自底向上法没有使用递归，因此具有更小的时间系数。

*钢铁分割（带备忘的自顶向下法）*
```java
public int memoied_cut_rod(int[] p, int n) {
    // r数组用来保存子问题的最优解
    int[] r = new int[n + 1];
    for (int i = 0; i < n + 1; i++) {
        r[i] = Integer.MIN_VALUE;
    }
    return memoied_cut_rod_aux(p, n, r);
}

public int memoied_cut_rod_aux(int[] p, int n, int[] r) {
    // 在备忘录中查到了子问题的最优解
    if (r[n] >= 0) {
        return r[n];
    }
    // 没有在备忘录中查到子问题的最优解
    int q = Integer.MIN_VALUE;
    // 问题求解完毕
    if (n == 0) {
        q = 0;
    }
    else {
        // 循环求解分割n的不同方式，第一次左边为1，右边为n-1,第二次左边为2，
        // 右边为n-2，求出他们之中的最大值
        for (int i = 0; i < Math.min(n, p.length); i++) {
            // 最优子结构
            q = Math.max(q, p[i] + memoied_cut_rod_aux(p, n - i - 1, r));
        }
    }
    // 求出了问题规模为n的最优解，保存子问题最优解
    r[n] = q;
    return q;
}
```

主过程`memoied_cut_rod`将辅助数组（备忘录）`r[0..n]`的元素均初始化为整型的负无穷，然后调用辅助过程`memoied_cut_rod_aux`。

过程`memoied_cut_rod_aux`引入了备忘机制，首先检查备忘录中是否含有所需值，如果有，则直接返回保存的值；否则按照通常的方法计算所需值`q`，最后将计算结果`q`保存在备忘录`r[n]`中，并且返回计算结果。

*钢条分割（自底向上法）*

```java
public int bottom_up_cut_rod(int[] p, int n) {
    // 辅助数组 r，用来记录子问题的最优解，最后一个元素即为规模为 n 的问题的最优解
    int[] r = new int[n + 1];
    for (int i = 0; i < n + 1; i++) {
        r[i] = Integer.MIN_VALUE;
    }
    r[0] = 0;
    int q = Integer.MIN_VALUE;
    // 首先从规模为 1 开始求解子问题的最优解，然后逐步扩大规模
    for (int i = 1; i <= n; i++) {
        // 求解规模为 i 的最优解
        // 将长度为 i 的钢条分割为长度为 j 的子钢条和长度为 i - j 的子钢条
        for (int j = 1; j <= Math.min(i, p.length); j++) {
            q = Math.max(q, p[j - 1] + r[i - j]);
        }
        r[i] = q;
    }
    return r[n];
}
```

自底向上版本`bottom_up_cut_rod`采用子问题的自然顺序：若`j < i`，则规模为`j`的子问题比规模为`i`的子问题“更小”。因此，过程逐次求解规模为`i = 0,1...n`的子问题。

过程`bottom_up_cut_rod`的首先创建一个新数组来保存子问题的解，并且将`r[0]`初始化为0，其他初始化为负无穷（整型）。然后逐次求解规模为`i`的子问题，求解规模为`i`的子问题的方法与带备忘录的方法相同，现在可以直接访问`r[i - j]`的来获得规模为`i - j`的子问题的解，不必递归调用。每次求解完一个子问题的最优解都要将其存储到数组`r[i]`中，最后数组的最后一个元素即为最优解。

自底向上算法和自顶向下算法具有相同的渐进运行时间。其时间复杂度为O（n^2）。

## 子问题图

当我们思考动态规划问题时，我们应该弄清楚所涉及的子问题以及子问题之间的关系。

子问题图G=（V, E）的规模可以帮助我们确定动态规划算法的运行时间。算法运行时间等于每个子问题求解时间之和。动态规划算法与顶点和边的数量呈线性关系。

## 重构解

钢条切割问题的动态规划算法返回的是最优解的收益值，但是没有返回解本身（一个长度列表，给出切割后每段钢条的长度）。我们可以扩展动态规划算法，使之对每个子问题不仅保存最优收益值，还保存对应的切割方案。

```java
public List<int[]> extended_bottom_up_cut_rod(int[] p, int n) {
    List<int[]> result = new ArrayList<>();
    int[] r = new int[n + 1], s = new int[n + 1];
    for (int i = 1; i <= n; i++) {
        r[i] = Integer.MIN_VALUE;
        s[i] = Integer.MIN_VALUE;
    }
    r[0] = 0;
    s[0] = 0;
    int q = Integer.MIN_VALUE;
    // 自底向上构建子问题的最优解
    for (int j = 1; j <= n; j++) {
        // 构建规模为j的最优解需要用到规模小于j的子问题的最优解
        for (int i = 1; i <= Math.min(j, p.length); i++) {
            if (q < p[i - 1] + r[j - i]) {
                q = p[i - 1] + r[j - i];
                // 存放长度为j的第一段切割长度为i
                s[j] = i;
            }
        }
        r[j] = q;
    }
    result.add(r);
    result.add(s);
    return result;
}
```

创建了数组`s`，并在求解规模为`j`的子问题时将第一段钢条的最优切割长度`i`保存在`s[j]`中。

输出最优切割方案：
```java
// 输出最优方案
public void print_cut_rod_solution(int[] p, int n) {
    List<int[]> result = this.extended_bottom_up_cut_rod(p, n);
    System.out.println("钢条切割最大价值为： " + result.get(0)[n]);
    System.out.print("钢条切割最佳方案为： ");
    while (n > 0) {
        System.out.print(result.get(1)[n]);
        System.out.print("\t");
        n = n - result.get(1)[n];
    }
}
```

## 矩阵链乘法

目标是确定代价最低的计算顺序。

*步骤一：最优括号化方案的结构特征（可以采用反证法和数学归纳法证明）*

动态规划方法的第一步是寻找最优子结构，然后就可以利用这种子结构从子问题的最优解构造原问题的最优解。

为了对`A i，A (i+1)...A j`进行括号化，我们就必须在某个`A k`和`A (k+1)`之间将矩阵链划分开。也就是说，我们首先计算矩阵`A i...k`和`A k+1...j`，然后再计算它们的乘积得到最终结果`A i...A j`。此方案的计算代价等于矩阵`A i...k`加上`A k+1...j`的计算代价，以及两者相乘的计算代价。

证明本问题的最优子结构。假设`A i，A (i+1)...A j`的最优化括号方案的分割点在`A k`和`A (k+1)`之间。在对`A i，A (i+1)...A k`进行括号化时，我们应该采用独立求解它时的最优方案。如果不采用独立求解的`A i，A (i+1)...A k`的最优方案来对它进行括号化，那么将此最优解带入`A i，A (i+1)...A j`的最优解中，`A i，A (i+1)...A j`的最优化括号方案的分割点就不在在`A k`和`A (k+1)`之间，与假设矛盾。

我们可以将问题划分为两个子问题，求出子问题实例的最优解，然后将子问题的最优解组合起来。在确定分割点时，必须保证已经考察了所有可能的划分点。

*步骤2：一个递归求解方案*

```java
// (伪代码)
// 递归求解公式的代价
// m[i, j] 表示计算矩阵A i...j 所需标量乘法的最小值
if (i == j) {
    m[i, j] = 0;
} else if (i < j) {
    m[i, j] = Math.min(m[i, k] + m[k+1, j] + pi-1 * pk * pj)
}
```
`m[i, j]`给出了子问题最优解的代价，但他并未提供足够的信息来构造最优解。我们用s[i, j]保存`A i，A (i+1)...A j`最优括号化方案的分割点位置k。
```java
public int dp_matrix_mutiply(int[] p) {
    int n = p.length - 1;
    // m[i, j]保存最有代价
    int[][] m= new int[n + 1][n + 1];
    // s[1...n-1, 2...n]保存切割点
    int[][] s = new int[n][n];
    for (int i = 1; i <= n; i++) {
        m[i][i] = 0;
    }
    return recursion_matrix_mutiply(p, m, s, 1, n);
}
public int recursion_matrix_mutiply(int[] p, int[][] m, int[][] s,
                                        int i, int j) {
    if (i == j) {
        return 0;
    }
    int q = Integer.MAX_VALUE;
    // 求解子问题 m[i, j]
    for (int k = i; k < j; k++) {
        m[i][k] = recursion_matrix_mutiply(p, m , s, i, k);
        m[k + 1][j] = recursion_matrix_mutiply(p, m, s, k + 1, j);
        int temp = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
        if (q > temp) {
            q = temp;
            m[i][j] = temp;
        }
    }
    return m[i][j];
}
```

*步骤三：计算最优代价*

自顶向下的动态规划算法（带备忘录）
```java
public int[][] dp_matrix_mutiply(int[] p) {
    long startTime = System.currentTimeMillis();    //获取开始时间
    int n = p.length - 1;
    // m[i, j]保存最有代价
    int[][] m= new int[n + 1][n + 1];
    // s[1...n-1, 2...n]保存切割点
    int[][] s = new int[n + 1][n + 1];
    for (int i = 1; i <= n; i++) {
        m[i][i] = 0;
    }
    recursion_matrix_mutiply(p, m, s, 1, n);
    long endTime = System.currentTimeMillis();    //获取结束时间
    System.out.println("程序运行时间：" + (endTime - startTime) + "ms");    //输出程序运行时间
    return m;
}
public int recursion_matrix_mutiply(int[] p, int[][] m, int[][] s,
                                        int i, int j) {
    if (i == j) {
        return 0;
    }
    if (m[i][j] > 0) {
        return m[i][j];
    }
    int q = Integer.MAX_VALUE;
    // 求解子问题 m[i, j]
    for (int k = i; k < j; k++) {
        m[i][k] = recursion_matrix_mutiply(p, m , s, i, k);
        m[k + 1][j] = recursion_matrix_mutiply(p, m, s, k + 1, j);
        int temp = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
        if (q > temp) {
            q = temp;
            m[i][j] = temp;
            s[i][j] = k;
        }
    }
    return m[i][j];
}
```

自底向上的动态规划算法

```java
public int bottom_up__matrix_mutiply(int[] p, int[][] m, int[][] s, int n) {
    // x表示规模
    for (int x = 2; x <= n; x++) {
        for (int i = 1; i <= n - x + 1; i++) {
            // 求解规模为m[i, j]的子问题
            int q = Integer.MAX_VALUE;
            int j = i + x - 1;
            for (int k = i; k < j; k++) {
                int temp = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
                if (q > temp) {
                    q = temp;
                    m[i][j] = temp;
                    s[i][j] = k;
                }
            }
        }
    }
    return m[1][n];
}
```
