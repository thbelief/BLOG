# 刷题

## 简介 <a href="#jian-jie" id="jian-jie"></a>

按照[CodeTop企业题库](https://codetop.cc/home)高频题降序刷。

## 排序 <a href="#pai-xu" id="pai-xu"></a>

```java
package com.test.Sort;

import java.util.Arrays;
import java.util.Queue;

public class Sort {
    //参考博客如下
    //https://program.blog.csdn.net/article/details/83785159
    /*
    * 冒泡排序
    * 步骤1: 比较相邻的元素。如果第一个比第二个大，就交换它们两个；
    * 步骤2: 对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对，这样在最后的元素应该会是最大的数；
    * 步骤3: 针对所有的元素重复以上的步骤，除了最后一个；
    * 步骤4: 重复步骤1~3，直到排序完成。
    */
    public static int[] maopao(int[] array){
        if(array.length<=1) return array;
        for(int i=0;i<array.length;i++){
            for(int j=0;j<array.length-1-i;j++){
                if(array[j+1]<array[j]){
                    int temp=array[j+1];
                    array[j+1]=array[j];
                    array[j]=temp;
                }
            }
        }
        return array;
    }

    /*
    * 选择排序
    * 首先在未排序序列中找到最小（大）元素，
    * 存放到排序序列的起始位置，然后，
    * 再从剩余未排序元素中继续寻找最小（大）元素，
    * 然后放到已排序序列的末尾。以此类推，
    * 直到所有元素均排序完毕。
    */
    public static int[] xuanze(int[] array){
        if(array.length<=1) return array;
        //标识 存放当前最小的数的索引
        int index=0;
        for(int i=0;i<array.length;i++){
            index=i;
            for(int j=i;j<array.length;j++){
                if(array[j]<array[index]) index=j;
            }
            int temp=array[i];
            array[i]=array[index];
            array[index]=temp;
        }
        return array;
    }

    /*
    * 插入排序
    * 步骤1: 从第一个元素开始，该元素可以认为已经被排序；
    * 步骤2: 取出下一个元素，在已经排序的元素序列中从后向前扫描；
    * 步骤3: 如果该元素（已排序）大于新元素，将该元素移到下一位置；
    * 步骤4: 重复步骤3，直到找到已排序的元素小于或者等于新元素的位置；
    * 步骤5: 将新元素插入到该位置后；
    * 步骤6: 重复步骤2~5。
    */
    public static int[] charu(int[] array){
        if(array.length<=1) return array;
        //存放当前的元素值
        int cur=0;
        //假设第一个已经排序了，排好序的后面一个取出来，然后
        //往前走，假如当前的值大于等于前面某一个的时候，依次后退一个，然后插入即可
        for(int i=0;i<array.length-1;i++){
            cur=array[i+1];
            int preIndex=i;
            while(preIndex>=0&&cur<array[preIndex]){
                array[preIndex+1]=array[preIndex];
                preIndex--;
            }
            array[preIndex+1]=cur;
        }
        return array;
    }

    /*
    * 归并排序
    * 把长度为n的输入序列分成两个长度为n/2的子序列；
    * 对这两个子序列分别采用归并排序；
    * 将两个排序好的子序列合并成一个最终的排序序列。
    */
    public static int[] guibing(int[] array){
        if(array.length<=1) return array;
        int mid=array.length/2;
        int[] left= Arrays.copyOfRange(array,0,mid);
        int[] right= Arrays.copyOfRange(array,mid,array.length);
        return merge(guibing(left),guibing(right));
    }
    //分治
    public static int[] merge(int[] left,int[] right){
        int[] result=new int[left.length+right.length];
        for(int index=0,i=0,j=0;index<result.length;index++){
            if(i>=left.length) result[index]=right[j++];
            else if(j>=right.length) result[index]=left[i++];
            else if(left[i]>right[j]) result[index]=right[j++];
            else result[index]=left[i++];
        }
        return result;
    }

    /*
    * 快速排序
    * 从数列中挑出一个元素，称为 “基准”（pivot ）；
    * 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作；
    * 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。
    */
    public static void swap(int[] array,int i,int j){
        int temp=array[i];
        array[i]=array[j];
        array[j]=temp;
    }
    /*
    * 自写快排
    * 思路：每次随机选择一个元素temp 通过哨兵left+right不断逼近
    * right从右向左找比temp小的 如果比temp小 停止 然后left开始从左向右寻找比temp大的 如果找到停止  交换right和left索引位置的数值
    * 如果重复，直到遇到left==right的时候代表的是已经找到temp应该在的位置了交换即可
    * 然后就是递归调用以上 操作即可
    */
    public static void QuickSortByTHBELIEF(int[] array,int start,int end){
        //出口条件 不需要再排了
        if(start>=end) return;
        int left=start,right=end;
        //必须随机
        int temp=start+(int)Math.random()*(end-start+1);
        //如果不相等 重复做逼近操作
        while(left<right){
            //如果大的话 直接略过
            while(right>left&&array[right]>=array[temp]) right--;
            //此时跳出while之后一定是小于等于temp索引的值的
            while(right>left&&array[left]<=array[temp]) left++;
            //此时将找到的值交换
            swap(array,left,right);
        }
        //left和right相等 则直接交换temp的值和left的值即可 也算是找到了temp所应该在的位置
        swap(array,temp,left);
        //对基准值左边的元素进行排序
        QuickSortByTHBELIEF(array,start,left-1);
        //对基准值右边的元素进行排序
        QuickSortByTHBELIEF(array,right+1,end);
    }
    public static void kuaisu(int[] array,int start,int end){
        //当不止一个数的时候
        if(start<end){
            //左右指针 基准数
            int temp=array[start];
            int i=start,j=end;
            while(i<j){
                //当右边数大于基准数的时候跳过 继续向左查找
                //不满足条件的时候跳出循环 此时的j对应的元素是小于基准元素的
                while(i<j&&array[j]>temp){
                    j--;
                }
                //将右边小于等于基准元素的数填入到左边相应位置
                swap(array,i,j);
                //当左边的数小于等于基准数时，略过，继续向右查找
                //(重复的基准元素集合到左区间)
                //不满足条件时跳出循环，此时的i对应的元素是大于等于基准元素的
                while(i<j&&array[i]<=temp){
                    i++;
                }
                swap(array,i,j);
            }
            //将基准元素填入相应位置
            array[i]=temp;
            //此时的i即为基准元素的位置
            //对基准元素的左边子区间进行相似的快速排序
            kuaisu(array,start,i-1);
            //对基准元素的右边子区间进行相似的快速排序
            kuaisu(array,i+1,end);
        }else{
            return ;
        }
    }

    /*
    * 堆排序
    * 将初始待排序关键字序列(R1,R2….Rn)构建成大顶堆，此堆为初始的无序区；
    * 将堆顶元素R[1]与最后一个元素R[n]交换，此时得到新的无序区(R1,R2,……Rn-1)和新的有序区(Rn),且满足R[1,2…n-1]<=R[n]；
    * 由于交换后新的堆顶R[1]可能违反堆的性质，因此需要对当前无序区(R1,R2,……Rn-1)调整为新堆，然后再次将R[1]与无序区最后一个元素交换，得到新的无序区(R1,R2….Rn-2)和新的有序区(Rn-1,Rn)。不断重复此过程直到有序区的元素个数为n-1，则整个排序过程完成。
    */
    //声明全局变量，用于记录数组array的长度；
    static int len;
    /**
     * 堆排序算法
     *
     * @param
     * @return
     */
    public static void HeapSort(int[] arr) {
        //1.构建大顶堆
        for(int i=arr.length/2-1;i>=0;i--){
            //从第一个非叶子结点从下至上，从右至左调整结构
            adjustHeap(arr,i,arr.length);
        }
        //2.调整堆结构+交换堆顶元素与末尾元素
        for(int j=arr.length-1;j>0;j--){
            swap(arr,0,j);//将堆顶元素与末尾元素进行交换
            adjustHeap(arr,0,j);//重新对堆进行调整
        }
    }
    /**
     * 调整大顶堆（仅是调整过程，建立在大顶堆已构建的基础上）
     * @param arr
     * @param i
     * @param length
     */
    public static void adjustHeap(int []arr,int i,int length){
        int temp = arr[i];//先取出当前元素i
        for(int k=i*2+1;k<length;k=k*2+1){//从i结点的左子结点开始，也就是2i+1处开始
            if(k+1<length && arr[k]<arr[k+1]){//如果左子结点小于右子结点，k指向右子结点
                k++;
            }
            if(arr[k] >temp){//如果子节点大于父节点，将子节点值赋给父节点（不用进行交换）
                arr[i] = arr[k];
                i = k;
            }else{
                break;
            }
        }
        arr[i] = temp;//将temp值放到最终的位置
    }


}
```

## DFS BFS <a href="#dfs-bfs" id="dfs-bfs"></a>

```java
//使用Queue实现BFS 
public void BFSWithQueue(TreeNode root) { 
    Queue<TreeNode> queue = new LinkedList<>(); 
    if (root != null) 
        queue.add(root); 
    while (!queue.isEmpty()) { 
        TreeNode treeNode = queue.poll(); 

        //在这里处理遍历到的TreeNode节点 

        if (treeNode.left != null) 
            queue.add(treeNode.left); 
        if (treeNode.right != null) 
            queue.add(treeNode.right); 
    } 
}


//DFS递归实现 
public void DFSWithRecursion(TreeNode root) { 
    if (root == null) 
        return; 

    //在这里处理遍历到的TreeNode节点 

    if (root.left != null) 
        DFSWithRecursion(root.left); 
    if (root.right != null) 
        DFSWithRecursion(root.right); 
}


//DFS的迭代实现版本（Stack） 
public void DFSWithStack(TreeNode root) { 
     if (root == null) 
         return; 
     Stack<TreeNode> stack = new Stack<>(); 
     stack.push(root); 

     while (!stack.isEmpty()) { 
         TreeNode treeNode = stack.pop(); 

         //在这里处理遍历到的TreeNode 

         if (treeNode.right != null) 
             stack.push(treeNode.right); 

         if (treeNode.left != null) 
             stack.push(treeNode.left); 
     } 
}
```

## [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/) <a href="#206-fan-zhuan-lian-biao" id="206-fan-zhuan-lian-biao"></a>

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        /**
        迭代
        每次循环将当前cur的next指向pre
        然后pre和cur依次后移一位
         */
        ListNode pre = null;
        while(head != null){
            ListNode temp = head.next;
            head.next = pre;
            pre = head;
            head = temp;
        }
        return pre;
        //return reverse(null,head);
    }

    /**
    递归
    先想好跳出递归的条件，无非是链表遍历完了
    然后是本次递归需要做的事情，首先是更新当前指针
    然后将当前指针的next指向pre，然后继续下一次递归即可
     */
    private ListNode reverse(ListNode prev, ListNode cur) {
        if (cur == null) {
            return prev;
        }
        ListNode temp = cur.next;
        cur.next = prev;// 反转
        // 更新prev、cur位置
        // prev = cur;
        // cur = temp;
        return reverse(cur, temp);
    }
}
```

## [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/) <a href="#3-wu-zhong-fu-zi-fu-de-zui-chang-zi-chuan" id="3-wu-zhong-fu-zi-fu-de-zui-chang-zi-chuan"></a>

```java
import java.util.*;
class Solution {
    public int lengthOfLongestSubstring(String s) {
        /**
        维护一个动态窗口
        每次max即可
        */
        int[] intArray = new int[128];
        Arrays.fill(intArray,0);
        int result = 0;
        int left=0,right = 0;
        char[] array = s.toCharArray();
        while(right < array.length){
            char c = array[right];
            right++;
            //右滑
            intArray[c]++;
            while(intArray[c]>=2){
                char d = array[left];
                left++;
                intArray[d]--;
            }
            result = Math.max(result,right-left);
        }
        return result;
    }
}
```

## [146. LRU 缓存](https://leetcode-cn.com/problems/lru-cache/) <a href="#146lru-huan-cun" id="146lru-huan-cun"></a>

```java
public class LRUCache {
    /**
    Hash定位以及前后继节点链表（双向链表）
    用hash表保证查找的时候O(1)
    同时双向链表保证刚使用过的到head 超过直接pop tail即可
    */
    class DLinkedNode {
        int key;
        int value;
        DLinkedNode prev;
        DLinkedNode next;
        public DLinkedNode() {}
        public DLinkedNode(int _key, int _value) {key = _key; value = _value;}
    }

    private Map<Integer, DLinkedNode> cache = new HashMap<Integer, DLinkedNode>();
    private int size;
    private int capacity;
    private DLinkedNode head, tail;

    public LRUCache(int capacity) {
        this.size = 0;
        this.capacity = capacity;
        // 使用伪头部和伪尾部节点
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        DLinkedNode node = cache.get(key);
        if (node == null) {
            return -1;
        }
        // 如果 key 存在，先通过哈希表定位，再移到头部
        moveToHead(node);
        return node.value;
    }

    public void put(int key, int value) {
        DLinkedNode node = cache.get(key);
        if (node == null) {
            // 如果 key 不存在，创建一个新的节点
            DLinkedNode newNode = new DLinkedNode(key, value);
            // 添加进哈希表
            cache.put(key, newNode);
            // 添加至双向链表的头部
            addToHead(newNode);
            ++size;
            if (size > capacity) {
                // 如果超出容量，删除双向链表的尾部节点
                DLinkedNode tail = removeTail();
                // 删除哈希表中对应的项
                cache.remove(tail.key);
                --size;
            }
        }
        else {
            // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            node.value = value;
            moveToHead(node);
        }
    }

    private void addToHead(DLinkedNode node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }

    private void removeNode(DLinkedNode node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void moveToHead(DLinkedNode node) {
        removeNode(node);
        addToHead(node);
    }

    private DLinkedNode removeTail() {
        DLinkedNode res = tail.prev;
        removeNode(res);
        return res;
    }
}
```

## [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/) <a href="#215-shu-zu-zhong-de-dikge-zui-da-yuan-su" id="215-shu-zu-zhong-de-dikge-zui-da-yuan-su"></a>

```java
public class Solution {
    /**
    使用类似于二分法的思路写外层
    内层参考快速排序的思路求k的位置
    */

    public int findKthLargest(int[] nums, int k) {
        int len = nums.length;
        int left = 0;
        int right = len - 1;

        // 转换一下，第 k 大元素的下标是 len - k
        int target = len - k;

        while (true) {
            int index = partition(nums, left, right);
            if (index == target) {
                return nums[index];
            } else if (index < target) {
                left = index + 1;
            } else {
                right = index - 1;
            }
        }
    }

    /**
     * 对数组 nums 的子区间 [left..right] 执行 partition 操作，返回 nums[left] 排序以后应该在的位置
     * 在遍历过程中保持循环不变量的定义：
     * nums[left + 1..j] < nums[left]
     * nums(j..i) >= nums[left]
     *
     * @param nums
     * @param left
     * @param right
     * @return
     */
    public int partition(int[] nums, int left, int right) {
        int pivot = nums[left];
        int j = left;
        for (int i = left + 1; i <= right; i++) {
            if (nums[i] < pivot) {
                // j 的初值为 left，先右移，再交换，小于 pivot 的元素都被交换到前面
                j++;
                swap(nums, j, i);
            }
        }
        // 在之前遍历的过程中，满足 nums[left + 1..j] < pivot，并且 nums(j..i) >= pivot
        swap(nums, j, left);
        // 交换以后 nums[left..j - 1] < pivot, nums[j] = pivot, nums[j + 1..right] >= pivot
        return j;
    }

    private void swap(int[] nums, int index1, int index2) {
        int temp = nums[index1];
        nums[index1] = nums[index2];
        nums[index2] = temp;
    }
}
```

## [25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/) <a href="#25k-ge-yi-zu-fan-zhuan-lian-biao" id="25k-ge-yi-zu-fan-zhuan-lian-biao"></a>

```java
class Solution {
    /**
    递归思想，每一层先处理该做的，就是翻转本层，然后将后续翻转的接到left即可
    */
    public ListNode reverseKGroup(ListNode head, int k) {
        if(head==null){
            return null;
        }
        ListNode left = head,right = head;
        // right代表的是第k个
        for(int i=0;i<k;i++){
            if(right==null){
                return head;
            }
            right = right.next;
        }
        // temp代表的是翻转后的链表
        ListNode temp = reverse(left,right);
        // 递归调用
        left.next = reverseKGroup(right,k);
        return temp;
    }

    /**
    翻转left到right的节点
    */
    public ListNode reverse(ListNode left,ListNode right){
        ListNode pre = null;
        while(left!=right){
            ListNode temp = left.next;
            left.next = pre;
            pre = left;
            left = temp;
        }
        return pre;
    }
}
```

## [15. 三数之和](https://leetcode-cn.com/problems/3sum/) <a href="#15-san-shu-zhi-he" id="15-san-shu-zhi-he"></a>

```java
class Solution {
    /**
    将三元组转换为二元组求解，注意边界条件以及重复问题（不让第一个重复即可）
    */
    public List<List<Integer>> threeSum(int[] nums) {
        // 一定要先排序
        Arrays.sort(nums);
        int len = nums.length;
        List<List<Integer>> result = new ArrayList();
        for(int i=0;i<len;i++){
            //将三元组转换为二元组
            List<List<Integer>> two = twoSum(nums,i+1,0-nums[i]);
            for(List<Integer> item:two){
                item.add(nums[i]);
                result.add(item);
            }
            // 关键在于提出重复数据，不让第一个数重复即可
            while(i<len-1&&nums[i]==nums[i+1]){
                i++;
            }
        }
        return result;
    }

    /**
    获取从start开始和为target的二元数组
    */
    public List<List<Integer>> twoSum(int[] list,int start,int target){
        int left = start,right = list.length-1;
        List<List<Integer>> result = new ArrayList();
        while(left<right){
            // sum求和
            int sum = list[left]+list[right];
            int curL = list[left],curR = list[right];
            if(sum<target){
                // list[left]==curL代表重合 跳过即可
                while(left<right&&list[left]==curL){
                    left++;
                }
            }else if(sum>target){
                while(left<right&&list[right]==curR){
                    right--;
                }
            }else{
                List<Integer> temp = new ArrayList();
                temp.add(list[left]);
                temp.add(list[right]);
                result.add(temp);
                // while是为了剔除重复项
                while(left<right&&list[left]==curL){
                    left++;
                }
                while(left<right&&list[right]==curR){
                    right--;
                }
            }
        }
        return result;
    }
}
```

## [912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/) <a href="#912-pai-xu-shu-zu" id="912-pai-xu-shu-zu"></a>

```java
import java.util.*;

class Solution {
    public int[] sortArray(int[] nums) {
        // 快速排序
        quickSort(nums,0,nums.length-1);
        return nums;
    }

    void quickSort(int[] nums, int start, int end) {
        if (start >= end) return;
        int left = start;
        int right = end;
        // 选取随机值，防止基本有序的数组时间复杂度由O(nlog2n)退化成O(n^2)
        swap(nums, start, (start + end) / 2);
        int target = nums[start];
        while (left < right) {
            // 先找小于基准值的元素，方便跳出循环时调整基准值的位置
            while (left < right && nums[right] > target) right--;
            // 再找大于等于基准值的元素（条件有等于是为了将基准值与刚才小于基准值的元素对调）
            while (left < right && nums[left] <= target) left++;
            if (left != right) swap(nums, left, right);
        }
        // 将基准值和 right 找到的比基准值小的元素对调
        // 对调后左边都比基准值小，右边都比基准值大
        if (right > start) swap(nums, right, start);
        quickSort(nums, start, right - 1);
        quickSort(nums, right + 1, end);
    }

    void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

## [53. 最大子数组和](https://leetcode-cn.com/problems/maximum-subarray/) <a href="#53-zui-da-zi-shu-zu-he" id="53-zui-da-zi-shu-zu-he"></a>

```java
class Solution {
    public int maxSubArray(int[] nums) {
        // dp[target]代表下标为target的之前连续子数组最大和
        int[] dp = new int[nums.length];
        // base case
        dp[0] = nums[0];
        int result = nums[0];
        for(int i=1;i<nums.length;i++){
            // 要么选择+当前的走连续 要么自己单独开始一个子数组
            dp[i]=Math.max(dp[i-1]+nums[i],nums[i]);
            result = Math.max(result,dp[i]);
        }
        return result;
    }
}
```

## [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/) <a href="#21-he-bing-liang-ge-you-xu-lian-biao" id="21-he-bing-liang-ge-you-xu-lian-biao"></a>

```java
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode pre = new ListNode();
        ListNode cur = pre;
        /**
        遍历，拼接，输出即可
        */
        while(list2!=null&&list1!=null){
            if(list2.val>list1.val){
                cur.next = list1;
                cur = cur.next;
                list1 = list1.next;
            }else{
                cur.next = list2;
                cur = cur.next;
                list2 = list2.next;
            }
        }
        if(list1!=null){
            cur.next=list1;
        }
        if(list2!=null){
            cur.next=list2;
        }
        return pre.next;
    }
}
```

## [1. 两数之和](https://leetcode-cn.com/problems/two-sum/) <a href="#1-liang-shu-zhi-he" id="1-liang-shu-zhi-he"></a>

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        /**
        可以使用hashmap去做
        */
        int[] res = new int[2];
        if(nums == null || nums.length == 0){
            return res;
        }
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < nums.length; i++){
            int temp = target - nums[i];
            if(map.containsKey(temp)){
                res[1] = i;
                res[0] = map.get(temp);
                return res;
            }
            map.put(nums[i], i);
        }
        return res;
    }
}
```

## [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/) <a href="#102-er-cha-shu-de-ceng-xu-bian-li" id="102-er-cha-shu-de-ceng-xu-bian-li"></a>

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        List<List<Integer>> result = new ArrayList();
        if(root!=null){
            queue.add(root);
        }
        /**
        使用队列，迭代实现BFS 然后通过count计算每层node个数
        控制循环次数，即可
         */
        while(!queue.isEmpty()){
            List<Integer> temp = new ArrayList();
            int count = queue.size();
            while(count!=0){
                TreeNode node = queue.poll();
                temp.add(node.val);
                if(node.left!=null){
                    queue.add(node.left);
                }
                if(node.right!=null){
                    queue.add(node.right);
                }
                count--;
            }
            result.add(temp);
        }
        return result;
    }
}
```

## [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/) <a href="#121-mai-mai-gu-piao-de-zui-jia-shi-ji" id="121-mai-mai-gu-piao-de-zui-jia-shi-ji"></a>

```java
class Solution {
    public int maxProfit(int[] prices) {
        // 第i天之前股票的最低价格
        int[] dp = new int[prices.length];
        dp[0] = prices[0];
        int result = 0;
        for(int i=1;i<prices.length;i++){
            // dp 思想 求的第i天之前最低价格
            dp[i] = Math.min(dp[i-1],prices[i]);
            // 比较当天-最低价格与之前的最大差价比较即可
            result = Math.max(result,prices[i] - dp[i]);
        }
        return result;
    }
}
```

## [141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/) <a href="#141-huan-xing-lian-biao" id="141-huan-xing-lian-biao"></a>

```java
public class Solution {
    /**
    快慢指针
    */
    public boolean hasCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        // 空链表、单节点链表一定不会有环
        while (fast != null && fast.next != null) {
            fast = fast.next.next; // 快指针，一次移动两步
            slow = slow.next;      // 慢指针，一次移动一步
            if (fast == slow) {   // 快慢指针相遇，表明有环
                return true;
            }
        }
        return false; // 正常走到链表末尾，表明没有环
    }
}
```

## [103. 二叉树的锯齿形层序遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/) <a href="#103-er-cha-shu-de-ju-chi-xing-ceng-xu-bian-li" id="103-er-cha-shu-de-ju-chi-xing-ceng-xu-bian-li"></a>

```java
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        /**
        方式很简单，bfs遍历，同时使用一个变量控制方向
        每下一层改变方向，同时注意每层界限划分
        */
        List<List<Integer>> res = new ArrayList<>();
        if (root == null)
            return res;
        //创建队列，保存节点
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);//先把节点加入到队列中
        boolean leftToRight = true;//第一步先从左边开始打印
        while (!queue.isEmpty()) {
            //记录每层节点的值
            List<Integer> level = new ArrayList<>();
            //统计这一层有多少个节点
            int count = queue.size();
            //遍历这一层的所有节点，把他们全部从队列中移出来，顺便
            //把他们的值加入到集合level中，接着再把他们的子节点（如果有）
            //加入到队列中
            for (int i = 0; i < count; i++) {
                //poll移除队列头部元素（队列在头部移除，尾部添加）
                TreeNode node = queue.poll();
                //判断是从左往右打印还是从右往左打印。
                if (leftToRight) {
                    //如果从左边打印，直接把访问的节点值加入到列表level的末尾即可
                    level.add(node.val);
                } else {
                    //如果是从右边开始打印，每次要把访问的节点值
                    //加入到列表的最前面
                    level.add(0, node.val);
                }
                //左右子节点如果不为空会被加入到队列中
                if (node.left != null)
                    queue.add(node.left);
                if (node.right != null)
                    queue.add(node.right);
            }
            //把这一层的节点值加入到集合res中
            res.add(level);
            //改变下次访问的方向
            leftToRight = !leftToRight;
        }
        return res;
    }
}
```

## [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/) <a href="#20-you-xiao-de-kuo-hao" id="20-you-xiao-de-kuo-hao"></a>

```java
class Solution {
    public boolean isValid(String s) {
        /**
        思路很简单，用栈来保存 先进后出
        遇到左边就入栈 遇到右边就出栈
        最后判断是否为空就行
        */
        if(s.isEmpty()){
            return true;
        }
        Stack<Character> stack = new Stack();
        for(char c : s.toCharArray()){
            if(c =='('){
                stack.push(')');
            } else if(c =='['){
                stack.push(']');
            } else if(c =='{'){
                stack.push('}');
            } else if(stack.empty()||stack.pop()!=c){
                return false;
            }
        }
        return stack.empty()?true:false;
    }
}
```

## [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/) <a href="#160-xiang-jiao-lian-biao" id="160-xiang-jiao-lian-biao"></a>

```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA==null||headB==null){
            return null;
        }
        ListNode nodeA = headA,nodeB = headB;
        /**
        思路很简单，当节点走到终点的时候直接转移到另外一个head，继续走，
        这样两个节点走的路程都是一样的，相遇的时候就是交点节点
         */
        while(nodeA!=nodeB){
            nodeA = nodeA == null ? headB : nodeA.next;
            nodeB = nodeB == null ? headA : nodeB.next;
        }
        return nodeA;
    }
}
```

## [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/) <a href="#236-er-cha-shu-de-zui-jin-gong-gong-zu-xian" id="236-er-cha-shu-de-zui-jin-gong-gong-zu-xian"></a>

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        /**
        1.p或q为root节点
        2.p、q分别在左右子树
        3.p、q在左子树，公共祖先递归在左子树中查找，每一层重复这几种检查
        4.p、q在右子树 同3
         */
        if (root == null) return null;
        // 如果p,q为根节点，则公共祖先为根节点
        if (root.val == p.val || root.val == q.val) return root;
        // 如果p,q在左子树，则公共祖先在左子树查找
        if (find(root.left, p) && find(root.left, q)) {
            return lowestCommonAncestor(root.left, p, q);
        }
        // 如果p,q在右子树，则公共祖先在右子树查找
        if (find(root.right, p) && find(root.right, q)) {
            return lowestCommonAncestor(root.right, p, q);
        }
        // 如果p,q分属两侧，则公共祖先为根节点
        return root;
    }

    /**
    是否在root节点的子节点中找到节点c
    */
    private boolean find(TreeNode root, TreeNode c) {
        if (root == null) return false;
        if (root.val == c.val) {
            return true;
        }

        return find(root.left, c) || find(root.right, c);
    }
}
```

## [88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/) <a href="#88-he-bing-liang-ge-you-xu-shu-zu" id="88-he-bing-liang-ge-you-xu-shu-zu"></a>

```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int p1 = 0, p2 = 0;
        int[] sorted = new int[m + n];
        int cur;
        int index = 0;
        /**
        利用有序列表这个特性
        p1和p2分别代表的是两个列表的头指针，指向当前列表的头
        然后取值比较即可
        需要注意的是边界值的取值时刻
         */
        while (p1 < m || p2 < n) {
            if (p1 == m) {
                cur = nums2[p2++];
            } else if (p2 == n) {
                cur = nums1[p1++];
            } else if (nums1[p1] < nums2[p2]) {
                cur = nums1[p1++];
            } else {
                cur = nums2[p2++];
            }
            sorted[index++] = cur;
        }
        for (int i = 0; i != m + n; ++i) {
            nums1[i] = sorted[i];
        }
    }
}
```

## [33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/) <a href="#33-sou-suo-xuan-zhuan-pai-xu-shu-zu" id="33-sou-suo-xuan-zhuan-pai-xu-shu-zu"></a>

```java
class Solution {
    public int search(int[] nums, int target) {
        /**
        如果中间的数小于最右边的数，则右半段是有序的，若中间数大于最右边数，则左半段是有序的，
        我们只要在有序的半段里用首尾两个数组来判断目标值是否在这一区域内，
        这样就可以确定保留哪半边了
        */
        int len = nums.length;
        int left = 0, right = len-1;
        while(left <= right){
            int mid = (left + right) / 2;
            if(nums[mid] == target)
                return mid;
            else if(nums[mid] < nums[right]){
                if(nums[mid] < target && target <= nums[right])
                    left = mid+1;
                else
                    right = mid-1;
            }
            else{
                if(nums[left] <= target && target < nums[mid])
                    right = mid-1;
                else
                    left = mid+1;
            }
        }
        return -1;
    }
}
```

## [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/) <a href="#5-zui-chang-hui-wen-zi-chuan" id="5-zui-chang-hui-wen-zi-chuan"></a>

```java
class Solution {
    // 主函数
    public String longestPalindrome(String s) {
        // 记录最长回文串
        String res = "";

        // 穷举以所有点（奇数一个点，偶数两个点）为中心的回文串
        for (int i = 0; i < s.length(); i++) {
            // 当回文串是奇数时，由一个中心点向两边扩散
            String s1 = palindrome(s, i, i);
            // 当回文串是偶数时，由中间的两个中心点向两边扩散
            String s2 = palindrome(s, i, i + 1);

            // 三元运算符：判断为真时取冒号前面的值，为假时取冒号后面的值
            res = res.length() > s1.length() ? res : s1;
            res = res.length() > s2.length() ? res : s2;
        }

        return res;
    }

    // 辅助函数：寻找回文串
    private String palindrome(String s, int left, int right) {
        // 在区间 [0, s.length() - 1] 中寻找回文串，防止下标越界
        while (left >=0 && right < s.length()) {
            // 是回文串时，继续向两边扩散
            if (s.charAt(left) == s.charAt(right)) {
                left--;
                right++;
            } else {
                break;
            }
        }

        // 循环结束时的条件是 s.charAt(left) != s.charAt(right), 所以正确的区间为 [left + 1, right), 方法 substring(start, end) 区间是 [start, end), 不包含 end
        return s.substring(left + 1, right);
    }
}
```

## [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/) <a href="#200-dao-yu-shu-liang" id="200-dao-yu-shu-liang"></a>

```java
class Solution {
    /**
    递归即可解决
    */
    public int numIslands(char[][] grid) {
        int islandNum = 0;
        /**
        遍历二维数组，遇到1则result+1同时递归将周边的数值感染为2
         */
        for(int i = 0; i < grid.length; i++){
            for(int j = 0; j < grid[0].length; j++){
                if(grid[i][j] == '1'){
                    infect(grid, i, j);
                    islandNum++;
                }
            }
        }
        return islandNum;
    }
    //感染函数
    public void infect(char[][] grid, int i, int j){
        if(i < 0 || i >= grid.length ||
           j < 0 || j >= grid[0].length || grid[i][j] != '1'){
            return;
        }
        grid[i][j] = '2';
        infect(grid, i + 1, j);
        infect(grid, i - 1, j);
        infect(grid, i, j + 1);
        infect(grid, i, j - 1);
    }
}
```

## [46. 全排列](https://leetcode-cn.com/problems/permutations/) <a href="#46-quan-pai-lie" id="46-quan-pai-lie"></a>

```java
class Solution {
    List<List<Integer>> result = new ArrayList();

    public List<List<Integer>> permute(int[] nums) {
        if(nums==null){
            return new ArrayList();
        }
        List<Integer> list = new ArrayList();
        backTrack(nums,list);
        return result;
    }

    /**
    使用回溯算法
    传入所有选择列表
    先判断是否满足条件，如果满足直接returnreturn
    遍历所有选择列表
    做选择
    递归backTrackbackTrack
    撤销选择
     */
    public void backTrack(int[] nums,List<Integer> list){
        if(list.size()==nums.length){
            result.add(new ArrayList(list));
        }
        for(int i=0;i<nums.length;i++){
            if(list.contains(nums[i])){
                continue;
            }
            list.add(nums[i]);
            backTrack(nums,list);
            list.remove(list.size() - 1);
        }
    }
}
```

## [415. 字符串相加](https://leetcode-cn.com/problems/add-strings/) <a href="#415-zi-fu-chuan-xiang-jia" id="415-zi-fu-chuan-xiang-jia"></a>

```java
class Solution {
    public String addStrings(String num1, String num2) {
        /**
        题目中明确说明，不能使用BigInteger以及转换为整数形式
        那么可以直接手动进位，每次使用%取得余数，添加，最后翻转即可
        注意 - ‘0’ 就直接变换为整数了
         */
        StringBuilder result = new StringBuilder();
        int carry = 0;
        int i = num1.length() - 1;
        int j = num2.length() - 1;
        while(i >= 0 || j >= 0 || carry != 0){
            if(i >= 0){
                carry += num1.charAt(i) - '0';
                i--;
            }
            if(j >= 0){
                carry += num2.charAt(j) - '0';
                j--;
            }
            result.append(carry%10);
            carry/=10;
        }
        return result.reverse().toString();
    }
}
```

## [142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/) <a href="#142-huan-xing-lian-biao-ii" id="142-huan-xing-lian-biao-ii"></a>

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        if(head==null){
            return null;
        }
        /**
        思路如下：
        先快慢指针找到是否有环，如果有的话，
        放一个other到head从头开始走，slow继续从相遇的节点走
        到最后相遇的时候就是入口所在
        可以直接画图
         */
        ListNode slow = head , fast = head;
        boolean hasCycle = false;
        while(fast.next != null && fast.next.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if(slow == fast){
                hasCycle = true;
                break;
            }
        }
        if(hasCycle){
            ListNode other = head;
            while(other != slow){
                slow = slow.next;
                other = other.next;
            }
            return slow;
        }
        return null;
    }
}
```

## [23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/) <a href="#23-he-bingkge-sheng-xu-lian-biao" id="23-he-bingkge-sheng-xu-lian-biao"></a>

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        /**
        可以直接放到最小堆中
        每次取最小的即可
         */
         if(lists==null||lists.length == 0){
             return null;
         }
        ListNode pre = new ListNode();
        ListNode cur = pre ;
        PriorityQueue<ListNode> pq = new PriorityQueue<>(new Comparator<ListNode>() {
            // o1代表右边 o2代表左边 本题中代表升序排列
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return o1.val - o2.val;
            }
        });
        for (ListNode list : lists) {
            if (list == null) {
                continue;
            }
            pq.add(list);
        }
        while(!pq.isEmpty()){
            ListNode node = pq.poll();
            cur.next = node;
            cur = cur.next;
            if(node.next !=null){
                pq.add(node.next);
            }
        }
        return pre.next;
    }
}
```

## [92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/) <a href="#92-fan-zhuan-lian-biao-ii" id="92-fan-zhuan-lian-biao-ii"></a>

```java
class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
       /**
        定位到要反转部分的头节点 2，head = 2；前驱结点 1，pre = 1；
        当前节点的下一个节点3调整为前驱节点的下一个节点 1->3->2->4->5,
        当前结点仍为2， 前驱结点依然是1，重复上一步操作。。。
        1->4->3->2->5.
       */
       ListNode node = new ListNode();
       node.next = head;
       ListNode pre = node;
       for(int i=1;i<left;i++){
           pre = pre.next;
       } 
       head = pre.next;
       for(int i = left ;i < right ;i++){
           ListNode next = head.next;
           head.next = next.next;
           next.next = pre.next;
           pre.next = next;
       }
       return node.next;
    }
}
```

## [54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/) <a href="#54-luo-xuan-ju-zhen" id="54-luo-xuan-ju-zhen"></a>

```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
         /**
         通过遍历完成一行或一列之后重新设定边界来排除重复
         纯模拟，需要仔细+细心
         */
         int left = 0 , right = matrix[0].length - 1;
         int top = 0 , down = matrix.length - 1;
         List<Integer> reuslt = new ArrayList();
         while(true){
             // 上
             for(int i = left ; i <= right ; i++){
                 reuslt.add(matrix[top][i]);
             }
             if(++top > down){
                 break;
             }
             // 下
             for(int i = top ; i <= down ; i++){
                 reuslt.add(matrix[i][right]);
             }
             if(--right < left){
                 break;
             }
             // 左
             for(int i = right ; i >= left ; i--){
                 reuslt.add(matrix[down][i]);
             }
             if(--down < top){
                 break;
             }
             // 右
             for(int i = down ; i >= top ; i--){
                 reuslt.add(matrix[i][left]);
             }
             if(++left > right){
                 break;
             }
         }
         return reuslt;
    }
}
```

## [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/) <a href="#300-zui-chang-di-zeng-zi-xu-lie" id="300-zui-chang-di-zeng-zi-xu-lie"></a>

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        /**
        动态规划
        dp[i]表示包含i位置在内的最长严格递增子序列的长度
         */
         int[] dp = new int[nums.length];
         Arrays.fill(dp,1);
         int result = 1;
         for(int i = 1;i < nums.length;i++){
             for(int j=0;j<i;j++){
                 if(nums[j]<nums[i]){
                    dp[i] = Math.max(dp[i],dp[j]+1);
                    result = Math.max(dp[i],result);
                 }
             }
         }
         return result;
    }
}
```

## [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/) <a href="#42-jie-yu-shui" id="42-jie-yu-shui"></a>

```java
class Solution {
    public int trap(int[] height) {
        /**
        动态规划
        dp[i]表示i左边最高的和右边最高中矮的那个
        也就是最终能承受住的雨水量
        时间复杂度和空间复杂度都是O n
        需要注意的就是i的位置左右高度都是不包括当前高度的
         */
         int result = 0;
         int[] maxLeft = new int[height.length];
         int[] maxRight = new int[height.length];
         for(int i = 1;i<height.length;i++){
             maxLeft[i] = Math.max(height[i-1],maxLeft[i-1]);
         }
         for(int i = height.length - 2;i>=0;i--){
             maxRight[i] = Math.max(height[i+1],maxRight[i+1]);
         }
         for(int i=1;i<height.length - 1 ;i++){
             int min = Math.min(maxLeft[i],maxRight[i]);
             if(min>height[i]){
                 result+=min -height[i];
             }
         }
         return result;
    }
}
```

## [704. 二分查找](https://leetcode-cn.com/problems/binary-search/) <a href="#704-er-fen-cha-zhao" id="704-er-fen-cha-zhao"></a>

```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0,right = nums.length - 1;
        // 注意这里是小于等于还是小于会决定right是否等于mid - 1
        while(left <= right){
            // 基于left的基础上加上right-left的一半即中间
            int mid = left + (right - left) / 2;
            if(nums[mid] == target){
                return mid;
            }else if(nums[mid] < target){
                left = mid + 1;
            } else{
                right = mid - 1;
            }
        }
        return -1;
    }
}
```

## [143. 重排链表](https://leetcode-cn.com/problems/reorder-list/) <a href="#143-zhong-pai-lian-biao" id="143-zhong-pai-lian-biao"></a>

```java
class Solution {
    public void reorderList(ListNode head) {
        /**
        1.首先想到的思路是，直接遍历一遍，用一个n的数组存放每个节点，
        然后直接在head后按照顺序排列即可
        O(n)时间复杂度 O(n)空间复杂度
        2.快慢指针，找到中点
        将后半段翻转，然后指定一个节点到head
        依次插入即可
        时间复杂度O(n) 空间复杂度O(1)
         */
         if(head == null || head.next == null){
             return;
         }
         ListNode slow = head , fast = head;
         // 找到中点
         while(fast.next!=null&&fast.next.next!=null){
             slow = slow.next;
             fast = fast.next.next;
         }
         ListNode reverseAfter = slow.next;
         slow.next = null;
         reverseAfter = reverse(reverseAfter);

         ListNode cur = head;
         while(cur != null && reverseAfter != null){
             /**
             插入
              */
             ListNode curAfter = reverseAfter;
             reverseAfter = reverseAfter.next;
             ListNode curNext = cur.next;
             curAfter.next = cur.next;
             cur.next = curAfter;
             cur = curNext;
         }
    }

    public ListNode reverse(ListNode node){
        ListNode pre = null;
        ListNode cur = node;
        while(cur != null){
            /**
            存放当前节点的下一个节点
            当前节点的下一个改为pre
            pre设置为当前cur节点
            cur节点设置为原cur的next节点
             */
            ListNode nextNode = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nextNode;
        }
        return pre;
    }
}
```

## [94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/) <a href="#94-er-cha-shu-de-zhong-xu-bian-li" id="94-er-cha-shu-de-zhong-xu-bian-li"></a>

```java
class Solution {
    private List<Integer> result = new ArrayList();
    public List<Integer> inorderTraversal(TreeNode root) {
        solve(root);
        return result;
    }

    public void solve(TreeNode root){
        /**
        中序遍历很简单，就是遍历左节点之后
        遍历右节点即可
         */
        if(root == null){
            return;
        }
        solve(root.left);
        result.add(root.val);
        solve(root.right);
    }
}
```

## [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/) <a href="#124-er-cha-shu-zhong-de-zui-da-lu-jing-he" id="124-er-cha-shu-zhong-de-zui-da-lu-jing-he"></a>

```java
class Solution {
    private int result = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        dfs(root);
        return result;
    }

    /**
    首先需要明确的是父节点不关注子节点怎么计算的它只在乎子节点能提供的最大路径和
    返回当前节点能返回的对于父节点提供的最大的路径和
     */
    private int dfs(TreeNode root){
        if(root == null ){
            return 0;
        }
        // 递归计算子节点left的路径和（需要与0比较，小于0则代表无效，还不如不加）
        int left = Math.max(0,dfs(root.left));
        int right = Math.max(0,dfs(root.right));
        // result总是root的左右节点之后+当前的val
        result = Math.max(result,root.val+left+right);
        int max = Math.max(root.val + left ,root.val + right);
        // 如果是小于0 直接无效，不走这条路 因为小于0直接是负收益了
        return max < 0?0:max;
    }
}
```

## [232. 用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/) <a href="#232-yong-zhan-shi-xian-dui-lie" id="232-yong-zhan-shi-xian-dui-lie"></a>

```java
class MyQueue {
    /**
    栈 先进后出
    队列 先进先出
    思路：
    输入输出分开栈
    输入直接压入输入栈input
    需要输出时将input的值转换到output中然后输出
     */
    private Stack<Integer> input;
    private Stack<Integer> output;

    public MyQueue() {
        input = new Stack();
        output = new Stack();
    }

    public void push(int x) {
        input.push(x);
    }

    public int pop() {
        transform();
        return output.pop();
    }

    public int peek() {
        transform();
        return output.peek();
    }

    public boolean empty() {
        return input.isEmpty() && output.isEmpty();
    }

    /**
    用于输入输出栈转换
     */
    private void transform(){
        if(output.isEmpty()){
            while(!input.isEmpty()){
                output.push(input.pop());
            }
        }
    }
}
```

## [199. 二叉树的右视图](https://leetcode-cn.com/problems/binary-tree-right-side-view/) <a href="#199-er-cha-shu-de-you-shi-tu" id="199-er-cha-shu-de-you-shi-tu"></a>

```java
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        /**
        层序遍历 BFS 寻找当前层最后一个元素
         */
        List<Integer> result = new ArrayList();
        if(root == null){
            return result;
        }
        Queue<TreeNode> queue = new LinkedList();
        queue.offer(root);
        while(!queue.isEmpty()){
            int size = queue.size();
            for(int i=0;i<size;i++){
                TreeNode cur = queue.poll();
                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }
                if(i == size - 1){
                    result.add(cur.val);
                }
            }
        }
        return result;
    }
}
```

## [70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/) <a href="#70-pa-lou-ti" id="70-pa-lou-ti"></a>

```java
class Solution {
    public int climbStairs(int n) {
        /**
        典型的动态规划
        n的结果可以由n-1和n-2方案之和决定（因为1、2阶梯）
         */
         if(n<2){
             return 1;
         }
         int[] dp = new int[n + 1];
         dp[1] = 1;
         dp[2] = 2;
         for(int i = 3 ; i <= n ; i++){
             dp[i] = dp[i-1] + dp[i-2];
         }
         return dp[n];
    }
}
```

## [19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/) <a href="#19-shan-chu-lian-biao-de-dao-shu-dinge-jie-dian" id="19-shan-chu-lian-biao-de-dao-shu-dinge-jie-dian"></a>

```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        /**
        直接使用快慢指针即可
         */
        ListNode slow = head,fast = head;
        while(n-->0){
            fast = fast.next;
        }
        // 有可能删除第一个
        if(fast == null){
            return head.next;
        }
        while(fast.next!=null){
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return head;
    }
}
```

## [56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/) <a href="#56-he-bing-qu-jian" id="56-he-bing-qu-jian"></a>

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        /**
        思路如下：
        首先按照起始位置排序，排序完成之后区间一定是连续的。
        之后是遍历区间，比较结束位置是否重叠，如果重叠更新即可
         */
         Arrays.sort(intervals,(o1,o2) -> o1[0] - o2[0]);
         int[][] result = new int[intervals.length][2];
         int index = -1;
         for(int[] interval:intervals){
             //如果是起始状态或者当前区间起始位置大于result最后区间的结束位置
             //则可以直接加入
            if(index == -1 || interval[0] > result[index][1]){
                result[++index] = interval;
            }else{
                result[index][1] = Math.max(result[index][1] , interval[1]);
            }
         }
         // result可能后面还有没用的
         return Arrays.copyOf(result,index+1);
    }
}
```

## [4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/) <a href="#4-xun-zhao-liang-ge-zheng-xu-shu-zu-de-zhong-wei-shu" id="4-xun-zhao-liang-ge-zheng-xu-shu-zu-de-zhong-wei-shu"></a>

```java
class Solution {
  public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        int left = (m + n + 1) / 2;
        int right = (m + n + 2) / 2;
        return (findKth(nums1, 0, nums2, 0, left) + findKth(nums1, 0, nums2, 0, right)) / 2.0;
    }
    //i: nums1的起始位置 j: nums2的起始位置
    public int findKth(int[] nums1, int i, int[] nums2, int j, int k){
        if( i >= nums1.length) return nums2[j + k - 1];//nums1为空数组
        if( j >= nums2.length) return nums1[i + k - 1];//nums2为空数组
        if(k == 1){
            return Math.min(nums1[i], nums2[j]);
        }
        int midVal1 = (i + k / 2 - 1 < nums1.length) ? nums1[i + k / 2 - 1] : Integer.MAX_VALUE;
        int midVal2 = (j + k / 2 - 1 < nums2.length) ? nums2[j + k / 2 - 1] : Integer.MAX_VALUE;
        if(midVal1 < midVal2){
            return findKth(nums1, i + k / 2, nums2, j , k - k / 2);
        }else{
            return findKth(nums1, i, nums2, j + k / 2 , k - k / 2);
        }        
    }
}
```

## [82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/) <a href="#82-shan-chu-pai-xu-lian-biao-zhong-de-zhong-fu-yuan-su-ii" id="82-shan-chu-pai-xu-lian-biao-zhong-de-zhong-fu-yuan-su-ii"></a>

```java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if(head == null){
            return null;
        }
        /**
        时间复杂度O(n) 空间复杂度 O(1)
         */
        ListNode dummy = new ListNode(0,head);
        ListNode cur =dummy;
        while(cur.next!=null&&cur.next.next!=null){
            // 如果相等，移动到直到不想等的位置
            if(cur.next.val == cur.next.next.val){
                int temp = cur.next.val;
                while(cur.next!=null&&cur.next.val==temp){
                    cur.next = cur.next.next;
                }
            }else{
                //如果不相等直接往下走即可
                cur = cur.next;
            }
        }
        return dummy.next;
    }
}
```

## [69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/) <a href="#69x-de-ping-fang-gen" id="69x-de-ping-fang-gen"></a>

```java
class Solution {
    public int mySqrt(int x) {
        /**
        二分查找是最简单的
        求k*k小于等于x的最大k值
        */
        int left = 0,right = x;
        int result = -1;
        while(left<=right){
            int mid = left + (right - left) / 2;
            // int数值够大的时候会越界
            if(((long)mid * mid) <= x){
                result = mid;
                left = mid + 1;
            }else{
                right = mid - 1;
            }
        }
        return result;
    }
}
```

## [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/) <a href="#72-bian-ji-ju-li" id="72-bian-ji-ju-li"></a>

```java
class Solution {
    public int minDistance(String word1, String word2) {
        /**
        动态规划
        dp[i][j]代表word1的前i个转换为word2的前j个需要的最少操作
        if word1[i]==word2[j] dp[i][j] = dp[i - 1][j - 1]
        否则取三种方式中最少操作的一种+1即可
        else dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1
        */
        int len1 = word1.length(),len2 = word2.length();
        int[][] dp = new int[len1 + 1][len2 + 1];
        //注意边界情况
        for(int i = 0 ;i<=len1;i++){
            dp[i][0] = i;
        }
        for(int i = 0 ;i<=len2;i++){
            dp[0][i] = i;
        }
        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
                }
            }
        }
        return dp[len1][len2];
    }
}
```

## [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/) <a href="#2-liang-shu-xiang-jia" id="2-liang-shu-xiang-jia"></a>

```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        /**
        依旧是操作进位即可
        */
        ListNode root = new ListNode(0);
        ListNode cursor = root;
        int carry = 0;
        while(l1 != null || l2 != null || carry != 0) {
            int l1Val = l1 != null ? l1.val : 0;
            int l2Val = l2 != null ? l2.val : 0;
            int sumVal = l1Val + l2Val + carry;
            carry = sumVal / 10;

            ListNode sumNode = new ListNode(sumVal % 10);
            cursor.next = sumNode;
            cursor = sumNode;

            if(l1 != null) l1 = l1.next;
            if(l2 != null) l2 = l2.next;
        }

        return root.next;
    }
}
```

## [8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/) <a href="#8-zi-fu-chuan-zhuan-huan-zheng-shu-atoi" id="8-zi-fu-chuan-zhuan-huan-zheng-shu-atoi"></a>

```java
class Solution {
    public int myAtoi(String str) {
        /**
        去掉前导空格trim
        处理正负
        识别数字 注意边界情况
        */
        char[] chars = str.toCharArray();
        int n = chars.length;
        int idx = 0;
        while (idx < n && chars[idx] == ' ') {
            // 去掉前导空格
            idx++;
        }
        if (idx == n) {
            //去掉前导空格以后到了末尾了
            return 0;
        }
        boolean negative = false;
        if (chars[idx] == '-') {
            //遇到负号
            negative = true;
            idx++;
        } else if (chars[idx] == '+') {
            // 遇到正号
            idx++;
        } else if (!Character.isDigit(chars[idx])) {
            // 其他符号
            return 0;
        }
        int ans = 0;
        while (idx < n && Character.isDigit(chars[idx])) {
            int digit = chars[idx] - '0';
            if (ans > (Integer.MAX_VALUE - digit) / 10) {
                // 本来应该是 ans * 10 + digit > Integer.MAX_VALUE
                // 但是 *10 和 + digit 都有可能越界，所有都移动到右边去就可以了。
                return negative? Integer.MIN_VALUE : Integer.MAX_VALUE;
            }
            ans = ans * 10 + digit;
            idx++;
        }
        return negative? -ans : ans;
    }
}
```

## [148. 排序链表](https://leetcode-cn.com/problems/sort-list/) <a href="#148-pai-xu-lian-biao" id="148-pai-xu-lian-biao"></a>

```java
class Solution {
    public ListNode sortList(ListNode head) {
        return mergeSort(head);
    }

    // 归并排序
    private ListNode mergeSort(ListNode head){
        // 如果没有结点/只有一个结点，无需排序，直接返回
        if (head==null||head.next==null) return head;
        // 快慢指针找出中位点
        ListNode slowp=head,fastp=head.next.next,l,r;
        while (fastp!=null&&fastp.next!=null){
            slowp=slowp.next;
            fastp=fastp.next.next;
        }
        // 对右半部分进行归并排序
        r=mergeSort(slowp.next);
        // 链表判断结束的标志：末尾节点.next==null
        slowp.next=null;
        // 对左半部分进行归并排序
        l=mergeSort(head);
        return mergeList(l,r);
    }
    // 合并链表
    private ListNode mergeList(ListNode l,ListNode r){
        // 临时头节点
        ListNode tmpHead=new ListNode(-1);
        ListNode p=tmpHead;
        while (l!=null&&r!=null){
            if (l.val<r.val){
                p.next=l;
                l=l.next;
            }else {
                p.next=r;
                r=r.next;
            }
            p=p.next;
        }
        p.next=l==null?r:l;
        return tmpHead.next;
    }
}
```

## [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/) <a href="#jian-zhi-offer22-lian-biao-zhong-dao-shu-dikge-jie-dian" id="jian-zhi-offer22-lian-biao-zhong-dao-shu-dikge-jie-dian"></a>

```java
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        /**
        快慢指针保持k的距离
        */
        if(head==null){
            return null;
        }
        ListNode slow = head,fast = head;
        while(k > 0){
            fast = fast.next;
            k--;
        }
        while(fast!=null){
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }
}
```

## [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/) <a href="#1143-zui-chang-gong-gong-zi-xu-lie" id="1143-zui-chang-gong-gong-zi-xu-lie"></a>

```java
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        /**
        定义 dp[i][j] 表示 text1[0:i-1] 和 text2[0:j-1] 的最长公共子序  列。 （注：text1[0:i-1] 表示的是 text1 的 第 0 个元素到第 i - 1 个元素，两端都包含）
         */
         int len1 = text1.length();
         int len2 = text2.length();
         int[][] dp = new int[len1 + 1][len2 + 1];
         for(int i = 1; i <= len1 ; i++){
             for(int j = 1; j <= len2; j++){
                 if(text1.charAt(i - 1) == text2.charAt(j - 1)){
                     dp[i][j] = dp[i - 1][j - 1] + 1;
                 }else{
                     dp[i][j] = Math.max(dp[i-1][j],dp[i][j-1]);
                 }
             }
         }
         return dp[len1][len2];
    }
}
```

## [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/) <a href="#22-kuo-hao-sheng-cheng" id="22-kuo-hao-sheng-cheng"></a>

```java
import java.util.ArrayList;
import java.util.List;

public class Solution {

    /**
    做加法通过dfs去遍历整个二叉树，将路径保存下来即可
    注意剪枝条件left < right 一般只会left >= right
    */

    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        // 特判
        if (n == 0) {
            return res;
        }

        dfs("", 0, 0, n, res);
        return res;
    }

    /**
     * @param curStr 当前递归得到的结果
     * @param left   左括号已经用了几个
     * @param right  右括号已经用了几个
     * @param n      左括号、右括号一共得用几个
     * @param res    结果集
     */
    private void dfs(String curStr, int left, int right, int n, List<String> res) {
        if (left == n && right == n) {
            res.add(curStr);
            return;
        }

        // 剪枝
        if (left < right) {
            return;
        }

        if (left < n) {
            dfs(curStr + "(", left + 1, right, n, res);
        }
        if (right < n) {
            dfs(curStr + ")", left, right + 1, n, res);
        }
    }
}
```

## [144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/) <a href="#144-er-cha-shu-de-qian-xu-bian-li" id="144-er-cha-shu-de-qian-xu-bian-li"></a>

```java
class Solution {
    public List<Integer> preorderTraversal(TreeNode head) {
        List<Integer> res = new ArrayList();
        if (head == null) {
            return res;
        }
        /**
        用栈迭代即可
         */
        Stack<TreeNode> stack = new Stack<>();
        stack.push(head);
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            res.add(node.val);
            if (node.right != null) {
                stack.push(node.right);
            }
            if (node.left != null) {
                stack.push(node.left);
            }
        }
        return res;
    }
}
```

## [31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/) <a href="#31-xia-yi-ge-pai-lie" id="31-xia-yi-ge-pai-lie"></a>

```java
class Solution {
    public void nextPermutation(int[] nums) {
        //必须 原地 修改，只允许使用额外常数空间。
        /**
        具体思路：
        从后往前 首先找到后一个大于前一个的情况，i-1就是需要交换的位置
        然后排序（方便找最小的数）
        然后从i开始递增找，找第一个大于i-1的数字，交换即可，
        最后如果找不到，则排序即可
        */
        int len = nums.length;
        for (int i = len - 1; i > 0; i--) {
            //从后往前先找出第一个相邻的后一个大于前一个情况，此时的i-1位置就是需要交换的位置
            if (nums[i] > nums[i - 1]) {
                //对i自己和之后的元素排序，[i,len)从小到大，第一个大于i-1位置的进行交换，那么就是下一个排列
                Arrays.sort(nums, i, len);
                for (int j = i; j <len; j++) {
                    if (nums[j] > nums[i - 1]) {
                        int temp = nums[j];
                        nums[j] = nums[i - 1];
                        nums[i - 1] = temp;
                        return;
                    }
                }
            }
        }
        Arrays.sort(nums);//最后3,2,1情况的下一个就是1,2,3要重新排列成最小的，这种情况上面的交换执行不了 
    }
 }
```

## [41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/) <a href="#41-que-shi-de-di-yi-ge-zheng-shu" id="41-que-shi-de-di-yi-ge-zheng-shu"></a>

```java
class Solution {
    /**
    遍历一次数组把大于等于1的和小于数组大小的值放到原数组对应位置，然后再遍历一次数组查当前下标是否和值对应，如果不对应那这个下标就是答案，否则遍历完都没出现那么答案就是数组长度加1。
    */
    public int firstMissingPositive(int[] nums) {
       /**
       相当于手写哈希
       把数值为i的数放到下标i-1的位置
       f(nums[i]) = nums[i] - 1
        */ 
        int len = nums.length;

        for (int i = 0; i < len; i++) {
            while (nums[i] > 0 && nums[i] <= len && nums[nums[i] - 1] != nums[i]) {
                // 满足在指定范围内、并且没有放在正确的位置上，才交换
                // 例如：数值 3 应该放在索引 2 的位置上
                swap(nums, nums[i] - 1, i);
            }
        }

        // [1, -1, 3, 4]
        for (int i = 0; i < len; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        // 都正确则返回数组长度 + 1
        return len + 1;
    }

    private void swap(int[] nums, int index1, int index2) {
        int temp = nums[index1];
        nums[index1] = nums[index2];
        nums[index2] = temp;
    }
}
```

## [151. 颠倒字符串中的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/) <a href="#151-dian-dao-zi-fu-chuan-zhong-de-dan-ci" id="151-dian-dao-zi-fu-chuan-zhong-de-dan-ci"></a>

```java
class Solution {
   /**
     * 不使用Java内置方法实现
     * <p>
     * 1.去除首尾以及中间多余空格
     * 2.反转整个字符串
     * 3.反转各个单词
     */
    public String reverseWords(String s) {
        // System.out.println("ReverseWords.reverseWords2() called with: s = [" + s + "]");
        // 1.去除首尾以及中间多余空格
        StringBuilder sb = removeSpace(s);
        // 2.反转整个字符串
        reverseString(sb, 0, sb.length() - 1);
        // 3.反转各个单词
        reverseEachWord(sb);
        return sb.toString();
    }

    private StringBuilder removeSpace(String s) {
        // System.out.println("ReverseWords.removeSpace() called with: s = [" + s + "]");
        int start = 0;
        int end = s.length() - 1;
        while (s.charAt(start) == ' ') start++;
        while (s.charAt(end) == ' ') end--;
        StringBuilder sb = new StringBuilder();
        while (start <= end) {
            char c = s.charAt(start);
            if (c != ' ' || sb.charAt(sb.length() - 1) != ' ') {
                sb.append(c);
            }
            start++;
        }
        // System.out.println("ReverseWords.removeSpace returned: sb = [" + sb + "]");
        return sb;
    }

    /**
     * 反转字符串指定区间[start, end]的字符
     */
    public void reverseString(StringBuilder sb, int start, int end) {
        // System.out.println("ReverseWords.reverseString() called with: sb = [" + sb + "], start = [" + start + "], end = [" + end + "]");
        while (start < end) {
            char temp = sb.charAt(start);
            sb.setCharAt(start, sb.charAt(end));
            sb.setCharAt(end, temp);
            start++;
            end--;
        }
        // System.out.println("ReverseWords.reverseString returned: sb = [" + sb + "]");
    }

    private void reverseEachWord(StringBuilder sb) {
        int start = 0;
        int end = 1;
        int n = sb.length();
        while (start < n) {
            while (end < n && sb.charAt(end) != ' ') {
                end++;
            }
            reverseString(sb, start, end - 1);
            start = end + 1;
            end = start + 1;
        }
    }
}
```

## [93. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses/) <a href="#93-fu-yuan-ip-di-zhi" id="93-fu-yuan-ip-di-zhi"></a>

```java
class Solution {
    List<String> result = new ArrayList<>();

    public List<String> restoreIpAddresses(String s) {
        if (s.length() > 12) return result; // 算是剪枝了
        backTrack(s, 0, 0);
        return result;
    }

    // startIndex: 搜索的起始位置， pointNum:添加逗点的数量
    private void backTrack(String s, int startIndex, int pointNum) {
        if (pointNum == 3) {// 逗点数量为3时，分隔结束
            // 判断第四段⼦字符串是否合法，如果合法就放进result中
            if (isValid(s,startIndex,s.length()-1)) {
                result.add(s);
            }
            return;
        }
        for (int i = startIndex; i < s.length(); i++) {
            if (isValid(s, startIndex, i)) {
                s = s.substring(0, i + 1) + "." + s.substring(i + 1);    //在str的后⾯插⼊⼀个逗点
                pointNum++;
                backTrack(s, i + 2, pointNum);// 插⼊逗点之后下⼀个⼦串的起始位置为i+2
                pointNum--;// 回溯
                s = s.substring(0, i + 1) + s.substring(i + 2);// 回溯删掉逗点
            } else {
                break;
            }
        }
    }

    // 判断字符串s在左闭⼜闭区间[start, end]所组成的数字是否合法
    private Boolean isValid(String s, int start, int end) {
        if (start > end) {
            return false;
        }
        if (s.charAt(start) == '0' && start != end) { // 0开头的数字不合法
            return false;
        }
        int num = 0;
        for (int i = start; i <= end; i++) {
            if (s.charAt(i) > '9' || s.charAt(i) < '0') { // 遇到⾮数字字符不合法
                return false;
            }
            num = num * 10 + (s.charAt(i) - '0');
            if (num > 255) { // 如果⼤于255了不合法
                return false;
            }
        }
        return true;
    }
}
```

## [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) <a href="#105-cong-qian-xu-yu-zhong-xu-bian-li-xu-lie-gou-zao-er-cha-shu" id="105-cong-qian-xu-yu-zhong-xu-bian-li-xu-lie-gou-zao-er-cha-shu"></a>

```java
class Solution {
    private Map<Integer, Integer> indexMap;

    public TreeNode myBuildTree(int[] preorder, int[] inorder, int preorder_left, int preorder_right, int inorder_left, int inorder_right) {
        if (preorder_left > preorder_right) {
            return null;
        }

        // 前序遍历中的第一个节点就是根节点
        int preorder_root = preorder_left;
        // 在中序遍历中定位根节点
        int inorder_root = indexMap.get(preorder[preorder_root]);

        // 先把根节点建立出来
        TreeNode root = new TreeNode(preorder[preorder_root]);
        // 得到左子树中的节点数目
        int size_left_subtree = inorder_root - inorder_left;
        // 递归地构造左子树，并连接到根节点
        // 先序遍历中「从 左边界+1 开始的 size_left_subtree」个元素就对应了中序遍历中「从 左边界 开始到 根节点定位-1」的元素
        root.left = myBuildTree(preorder, inorder, preorder_left + 1, preorder_left + size_left_subtree, inorder_left, inorder_root - 1);
        // 递归地构造右子树，并连接到根节点
        // 先序遍历中「从 左边界+1+左子树节点数目 开始到 右边界」的元素就对应了中序遍历中「从 根节点定位+1 到 右边界」的元素
        root.right = myBuildTree(preorder, inorder, preorder_left + size_left_subtree + 1, preorder_right, inorder_root + 1, inorder_right);
        return root;
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int n = preorder.length;
        // 构造哈希映射，帮助我们快速定位根节点
        indexMap = new HashMap<Integer, Integer>();
        for (int i = 0; i < n; i++) {
            indexMap.put(inorder[i], i);
        }
        return myBuildTree(preorder, inorder, 0, n - 1, 0, n - 1);
    }
}
```

## [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/) <a href="#239-hua-dong-chuang-kou-zui-da-zhi" id="239-hua-dong-chuang-kou-zui-da-zhi"></a>

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int len = nums.length;
        /**
        使用双端队列
        变量的最前端（也就是 window.front()）是此次遍历的最大值的下标
        当我们遇到新的数时，将新的数和双项队列的末尾（也就是window.back()）
        比 较，如 果末尾比新数小，则把末尾扔掉，直到该队列的末尾比新数大或者
        队列为空的时候才停止，做法有点像使用栈进行括号匹配。
        双项队列中的所有值都要在窗口范围内
        */
        Deque<Integer> deque = new LinkedList();
        for(int i=0;i<k;i++){
            // 主要就是while 会保证deque的队首是最大的
            while(!deque.isEmpty()&&nums[i]>=nums[deque.peekLast()]){
                deque.pollLast();
            }
            deque.offerLast(i);
        }
        int[] result = new int[len-k+1];
        result[0] = nums[deque.peekFirst()];
        for(int i=k;i<len;i++){
            while(!deque.isEmpty()&&nums[i]>=nums[deque.peekLast()]){
                deque.pollLast();
            }
            deque.offerLast(i);
            while(deque.peekFirst()<=i-k){
                deque.pollFirst();
            }
            result[i-k+1] = nums[deque.peekFirst()];
        }
        return result;
    }
}
```

## [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/) <a href="#76-zui-xiao-fu-gai-zi-chuan" id="76-zui-xiao-fu-gai-zi-chuan"></a>

```java
class Solution {
    /**
    滑动窗口
    */
    public String minWindow(String s, String t) {
        if (s == null || s.length() == 0 || t == null || t.length() == 0){
            return "";
        }
        int[] need = new int[128];
        //记录需要的字符的个数
        for (int i = 0; i < t.length(); i++) {
            need[t.charAt(i)]++;
        }
        //l是当前左边界，r是当前右边界，size记录窗口大小，count是需求的字符个数，start是最小覆盖串开始的index
        int l = 0, r = 0, size = Integer.MAX_VALUE, count = t.length(), start = 0;
        //遍历所有字符
        while (r < s.length()) {
            char c = s.charAt(r);
            if (need[c] > 0) {//需要字符c
                count--;
            }
            need[c]--;//把右边的字符加入窗口
            if (count == 0) {//窗口中已经包含所有字符
                while (l < r && need[s.charAt(l)] < 0) {
                    need[s.charAt(l)]++;//释放右边移动出窗口的字符
                    l++;//指针右移
                }
                if (r - l + 1 < size) {//不能右移时候挑战最小窗口大小，更新最小窗口开始的start
                    size = r - l + 1;
                    start = l;//记录下最小值时候的开始位置，最后返回覆盖串时候会用到
                }
                //l向右移动后窗口肯定不能满足了 重新开始循环
                need[s.charAt(l)]++;
                l++;
                count++;
            }
            r++;
        }
        return size == Integer.MAX_VALUE ? "" : s.substring(start, start + size);
    }
}
```

## [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/) <a href="#110-ping-heng-er-cha-shu" id="110-ping-heng-er-cha-shu"></a>

```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        /**
        dfs+剪枝
        1.left-right<2 返回Math.max(left,right)+1
        2.left-right>=2 返回-1 即退出
        */
        return dfs(root) != -1;
    }

    public int dfs(TreeNode root){
        if(root == null){
            return 0;
        }
        int left = dfs(root.left);
        if(left == -1){
            return -1;
        }
        int right = dfs(root.right);
        if(right == -1){
            return -1;
        }
        return Math.abs(left-right)>=2?-1:Math.max(left,right)+1;
    }
}
```

## [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/) <a href="#104-er-cha-shu-de-zui-da-shen-du" id="104-er-cha-shu-de-zui-da-shen-du"></a>

```java
class Solution {
    public int maxDepth(TreeNode root) {
        /**
        DFS
        */
        return dfs(root);
    }

    public int dfs(TreeNode root){
        if(root == null){
            return 0;
        }
        int left = dfs(root.left);
        int right = dfs(root.right);
        return Math.max(left,right) + 1;
    }
}
```

## [129. 求根节点到叶节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/) <a href="#129-qiu-gen-jie-dian-dao-ye-jie-dian-shu-zi-zhi-he" id="129-qiu-gen-jie-dian-dao-ye-jie-dian-shu-zi-zhi-he"></a>

```java
class Solution {
    public int sumNumbers(TreeNode root) {
        /**
        dfs
        依次往下计算，直到叶子节点的时候再返回
        最后加left+right即可
        */
        return helper(root, 0);
    }

    public int helper(TreeNode root, int i){
        if (root == null) return 0;
        int temp = i * 10 + root.val;
        if (root.left == null && root.right == null)
            return temp;
        return helper(root.left, temp) + helper(root.right, temp);
    }
}
```

## [155. 最小栈](https://leetcode-cn.com/problems/min-stack/) <a href="#155-zui-xiao-zhan" id="155-zui-xiao-zhan"></a>

```java
class MinStack {
    /**
    在常数时间内检索到最小元素的栈。
    用个辅助栈 存放每个节点对应的最小节点即可
    mStack与mMinStack需要同步操作
    */
    Deque<Integer> xStack;
    Deque<Integer> minStack;
    public MinStack() {
        xStack = new LinkedList<Integer>();
        minStack = new LinkedList<Integer>();
        minStack.push(Integer.MAX_VALUE);
    }

    public void push(int x) {
        xStack.push(x);
        minStack.push(Math.min(minStack.peek(), x));
    }

    public void pop() {
        xStack.pop();
        minStack.pop();
    }

    public int top() {
        return xStack.peek();
    }

    public int getMin() {
        return minStack.peek();
    }
}
```

## [543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/) <a href="#543-er-cha-shu-de-zhi-jing" id="543-er-cha-shu-de-zhi-jing"></a>

```java
class Solution {
    int ans;
    /**
    其实就是dfs 求left+right+1的最大值 -1返回即可
    */
    public int diameterOfBinaryTree(TreeNode root) {
        ans = 1;
        depth(root);
        return ans - 1;
    }
    public int depth(TreeNode node) {
        if (node == null) {
            return 0; // 访问到空节点了，返回0
        }
        int L = depth(node.left); // 左儿子为根的子树的深度
        int R = depth(node.right); // 右儿子为根的子树的深度
        ans = Math.max(ans, L+R+1); // 计算d_node即L+R+1 并更新ans
        return Math.max(L, R) + 1; // 返回该节点为根的子树的深度
    }
}
```

## [32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/) <a href="#32-zui-chang-you-xiao-kuo-hao" id="32-zui-chang-you-xiao-kuo-hao"></a>

```java
class Solution {
    public int longestValidParentheses(String s) {
        /**
        用栈模拟 将无法匹配的括号的index标记1
        最后就转换为了最长连续问题
        */
        int result = 0;
        Stack<Integer> stack = new Stack();
        int[] mark = new int[s.length()];
        Arrays.fill(mark,0);
        for(int i = 0; i < s.length(); i++) {
            if(s.charAt(i) == '('){
                stack.push(i);
            } else {
                // 多余的右括号是不需要的，标记
                if(stack.empty()){
                    mark[i] = 1;
                }else{
                    stack.pop();
                }
            }
        }
        // 未匹配的左括号
        while(!stack.empty()){
            mark[stack.peek()] = 1;
            stack.pop();
        }
        // 寻找标记与标记之间的最大长度
        int len = 0;
        for(int i = 0; i < s.length(); i++) {
            if(mark[i]==1) {
                len = 0;
                continue;
            }
            len++;
            result = Math.max(result, len);
        }
        return result;
    }
}
```

## [98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/) <a href="#98-yan-zheng-er-cha-sou-suo-shu" id="98-yan-zheng-er-cha-sou-suo-shu"></a>

```java
class Solution {
    /**
    节点的左子树只包含 小于 当前节点的数。
    节点的右子树只包含 大于 当前节点的数。
    所有左子树和右子树自身必须也是二叉搜索树。
    dfs+边界更新
    */
    public boolean isValidBST(TreeNode root) {
        // 卡边界值
        return dfs(root,Long.MIN_VALUE,Long.MAX_VALUE);
    }

    public boolean dfs(TreeNode node,long min,long max){
        if(node == null){
            return true;
        }
        if(node.val <= min || node.val >= max){
            return false;
        }
        // 上界下界需要不断更新
        return dfs(node.left,min,node.val) && dfs(node.right,node.val,max);
    }
}
```

## [113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/) <a href="#113-lu-jing-zong-he-ii" id="113-lu-jing-zong-he-ii"></a>

```java
class Solution {
    private List<List<Integer>> result = new ArrayList();
    /**
    DFS 求所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
    */

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        dfs(root,targetSum,new ArrayList(),0);
        return result;
    }

    public void dfs(TreeNode root, int targetSum,List<Integer> path,int curSum){
        if(root == null){
            return;
        }
         if(root.left == null && root.right == null && root.val + curSum == targetSum){
            path.add(root.val);
            result.add(new ArrayList<>(path));
            path.remove(path.size() - 1);
            return ;
        }       
        path.add(root.val);
        curSum += root.val;
        dfs(root.left,targetSum,path,curSum);
        dfs(root.right,targetSum,path,curSum);
        path.remove(path.size()-1);
    }
}
```

## [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/) <a href="#101-dui-cheng-er-cha-shu" id="101-dui-cheng-er-cha-shu"></a>

```java
class Solution {
    /**
    dfs 直接比较left与right可以。
    bfs 也可以直接比较是否回文
    */
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return cmp(root.left, root.right);
    }

    private boolean cmp(TreeNode node1, TreeNode node2) {
        if (node1 == null && node2 == null) {
            return true;
        }
        if (node1 == null || node2 == null || node1.val != node2.val) {
            return false;
        }
        // left right互换
        return cmp(node1.left, node2.right) && cmp(node1.right, node2.left);
    }
}
```

## [165. 比较版本号](https://leetcode-cn.com/problems/compare-version-numbers/) <a href="#165-bi-jiao-ban-ben-hao" id="165-bi-jiao-ban-ben-hao"></a>

```java
class Solution {
    public int compareVersion(String v1, String v2) {
        /**
        用split加转义字符\\划分开之后，依次比较
        Integer.parseInt
        */
        String[] ss1 = v1.split("\\."), ss2 = v2.split("\\.");
        int n = ss1.length, m = ss2.length;
        int i = 0, j = 0;
        while (i < n || j < m) {
            int a = 0, b = 0;
            if (i < n) a = Integer.parseInt(ss1[i++]);
            if (j < m) b = Integer.parseInt(ss2[j++]);
            if (a != b) return a > b ? 1 : -1;
        }
        return 0;
    }
}
```

## [78. 子集](https://leetcode-cn.com/problems/subsets/) <a href="#78-zi-ji" id="78-zi-ji"></a>

```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        //可以使用回溯的算法
        /**
        以[]为基准，遍历nums的同时，继续增加新的即可
        */
        List<List<Integer>> result = new ArrayList();
        result.add(new ArrayList());
        for(int i = 0; i < nums.length ; i++){
            int size = result.size();
            for(int j = 0; j < size ;j++){
                List<Integer> temp = new ArrayList(result.get(j));
                temp.add(nums[i]);
                result.add(temp);
            }
        }
        return result;
    }
}
```

## [470. 用 Rand7() 实现 Rand10()](https://leetcode-cn.com/problems/implement-rand10-using-rand7/) <a href="#470-yong-rand7-shi-xian-rand10" id="470-yong-rand7-shi-xian-rand10"></a>

```java
/**
 * The rand7() API is already defined in the parent class SolBase.
 * public int rand7();
 * @return a random integer in the range 1 to 7
 */
class Solution extends SolBase {
    /**
    所谓均匀
     */
    public int rand10() {
        // 首先得到一个数
        int num = (rand7() - 1) * 7 + rand7();
        // 只要它还大于40，那你就给我不断生成吧
        while (num > 40)
            num = (rand7() - 1) * 7 + rand7();
        // 返回结果，+1是为了解决 40%10为0的情况
        return 1 + num % 10;
    }
}
```

## [43. 字符串相乘](https://leetcode-cn.com/problems/multiply-strings/) <a href="#43-zi-fu-chuan-xiang-cheng" id="43-zi-fu-chuan-xiang-cheng"></a>

```java
class Solution {
    public String multiply(String num1, String num2) {
        //不能使用任何内置的 BigInteger 库或直接将输入转换为整数。
        /**
        直接用数组存放相应位置获得的乘积，暂时不用进位，
        全部完成之后再依次进位即可
         */
        if(num1.equals("0") || num2.equals("0")){
            return "0";
        }
        int len1 = num1.length();
        int len2 = num2.length();
        int[] result = new int[len1 + len2];
        for(int i = len1 - 1; i >= 0; i--){
            int temp1 = num1.charAt(i) - '0';
            for(int j = len2 - 1; j >= 0; j--){
                int temp2 = num2.charAt(j) - '0';
                result[i+j+1] += temp1*temp2;
            }
        }
        for(int i = len1 + len2 -1; i > 0;i--){
            result[i-1] += result[i] / 10;
            result[i] %= 10;
        }
        int index = result[0] == 0 ?1:0;
        StringBuilder sb = new StringBuilder();
        while(index < len2 + len1){
            sb.append(result[index++]);
        }
        return sb.toString();
    }
}
```

## [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/) <a href="#322-ling-qian-dui-huan" id="322-ling-qian-dui-huan"></a>

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        /**
        动态规划
        dp[i]代表的是金额为i时最小硬币数
        填充max是为了方便确认-1
        */
        int max = amount + 1;
        int[] dp = new int[max];
        Arrays.fill(dp,max);
        dp[0] = 0;
        for(int i = 0 ; i <= amount ;i++){
            for(int j = 0;j<coins.length;j++){
                if(coins[j] <= i){
                    dp[i] = Math.min(dp[i],dp[i-coins[j]]+1);
                }
            }
        }
        return dp[amount] == max?-1:dp[amount];
    }
}
```

## [64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/) <a href="#64-zui-xiao-lu-jing-he" id="64-zui-xiao-lu-jing-he"></a>

```java
class Solution {
    public int minPathSum(int[][] grid) {
        // 每次只能向下或者向右移动一步。
        // 还是动态规划dp方程即可
        int lenX = grid[0].length;
        int lenY = grid.length;
        for(int i = 0;i<lenY;i++){
            for(int j = 0;j<lenX;j++){
                if(i == 0 && j == 0){
                    continue;
                } else if(j == 0){
                    grid[i][j] += grid[i - 1][j];
                } else if(i == 0){
                    grid[i][j] += grid[i][j - 1];
                } else{
                    grid[i][j] += Math.min(grid[i-1][j],grid[i][j-1]);
                }
            }
        }
        return grid[lenY - 1][lenX - 1];
    }
}
```

## [234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/) <a href="#234-hui-wen-lian-biao" id="234-hui-wen-lian-biao"></a>

```java
class Solution {
    public boolean isPalindrome(ListNode head) {
        //O(n) 时间复杂度和 O(1) 空间复杂度
        /**
        如果是回文链表，快慢指针找到中点，然后翻转后半部分，依次找即可知道是否是回文链表了。
        */
        if(head == null || head.next == null){
            return true;
        }
        ListNode slow = head,fast = head;
        while(fast.next != null && fast.next.next != null){
            fast = fast.next.next;
            slow = slow.next;
        }
        slow = reverse(slow.next);
        while(slow != null){
            if(head.val != slow.val){
                return false;
            }
            head = head.next;
            slow = slow.next;
        }
        return true;
    }

    /**
    翻转链表
     */
    public ListNode reverse(ListNode head){
        // 递归到最后一个节点，返回新的新的头结点
        if(head.next == null){
            return head;
        }
        ListNode newHead = reverse(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }
}
```

## [112. 路径总和](https://leetcode-cn.com/problems/path-sum/) <a href="#112-lu-jing-zong-he" id="112-lu-jing-zong-he"></a>

```java
class Solution {
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null) {
            return sum - root.val == 0;
        }
        return hasPathSum(root.left, sum - root.val)
                || hasPathSum(root.right, sum - root.val);
    }
}
```

## [718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/) <a href="#718-zui-chang-zhong-fu-zi-shu-zu" id="718-zui-chang-zhong-fu-zi-shu-zu"></a>

```java
class Solution {
    public int findLength(int[] nums1, int[] nums2) {
        int result = 0;
        int[][] dp = new int[nums1.length + 1][nums2.length + 1];

        //dp[i][j] ：以下标i - 1为结尾的A，和以下标j - 1为结尾的B，最长重复子数组长度为dp[i][j]。
        for (int i = 1; i < nums1.length + 1; i++) {
            for (int j = 1; j < nums2.length + 1; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    result = Math.max(result, dp[i][j]);
                }
            }
        }

        return result;
    }
}
```

## [169. 多数元素](https://leetcode-cn.com/problems/majority-element/) <a href="#169-duo-shu-yuan-su" id="169-duo-shu-yuan-su"></a>

```java
class Solution {
    public int majorityElement(int[] nums) {
        // 超过一半
        // 时间复杂度为 O(n)、空间复杂度为 O(1)
        // 摩尔投票法 对拼消耗 超过一半的一定不会输。
        int count = 1;
        int result = nums[0];
        for(int i=1;i<nums.length;i++){
            if(result == nums[i]){
                count++;
            }else{
                count--;
                if(count==0){
                    result = nums[i+1];
                }
            }
        }
        return result;
    }
}
```

## [48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/) <a href="#48-xuan-zhuan-tu-xiang" id="48-xuan-zhuan-tu-xiang"></a>

```java
class Solution {
    public void rotate(int[][] matrix) {
        // 要求 : 请不要 使用另一个矩阵来旋转图像
        // 先上下翻转再对角翻转
        int len = matrix.length;
        for(int i=0;i<len/2;i++){
            swap(matrix,i,len-1-i);
        }
        for(int i = 0; i < len; i++) {
            for(int j = i; j < len; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }

    public void swap(int[][] array,int a,int b){
        int[] temp = array[a];
        array[a] = array[b];
        array[b] = temp;
    }
}
```

## [226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/) <a href="#226-fan-zhuan-er-cha-shu" id="226-fan-zhuan-er-cha-shu"></a>

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        dfs(root);
        return root;
    }

    /**
    dfs 交换左右子节点即可
    */
    public void dfs(TreeNode root){
        if(root == null){
            return;
        }
        TreeNode tempLeft = root.left;
        root.left = root.right;
        root.right = tempLeft;
        dfs(root.left);
        dfs(root.right);
    }
}
```

## [39. 组合总和](https://leetcode-cn.com/problems/combination-sum/) <a href="#39-zu-he-zong-he" id="39-zu-he-zong-he"></a>

```java
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        // 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 
        // 考虑回溯
        List<List<Integer>> result = new ArrayList();
        backTrack(candidates,target,result,0,new ArrayList<Integer>());
        return result;
    }

    private void backTrack(int[] candidates, int target, List<List<Integer>> res, int i, ArrayList<Integer> tmp_list) {
        if(target < 0 ){
            return;
        } else if(target == 0){
            res.add(new ArrayList(tmp_list));
        }
        for(int start = i;start<candidates.length;start++){
            if(target<0){
                break;
            }
            tmp_list.add(candidates[start]);
            backTrack(candidates,target-candidates[start],res,start,tmp_list);
            tmp_list.remove(tmp_list.size()-1);
        }
    }
}
```

## [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/) <a href="#34-zai-pai-xu-shu-zu-zhong-cha-zhao-yuan-su-de-di-yi-ge-he-zui-hou-yi-ge-wei-zhi" id="34-zai-pai-xu-shu-zu-zhong-cha-zhao-yuan-su-de-di-yi-ge-he-zui-hou-yi-ge-wei-zhi"></a>

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        // 时间复杂度为 O(log n) 二分查找
        // 先找到一个 然后左右扩散
        int left = 0,right = nums.length - 1;
        while(left <= right){
            int mid = left + (right - left)/2;
            if (nums[mid] < target){
                left = mid + 1;
            } else if (nums[mid] > target){
                right = mid - 1;
            } else {
                // 找到下标为mid的值为target左右扩散
                return find(nums,mid);
            }
        }
        return new int[]{-1,-1};
    }

    public int[] find(int[] nums,int index){
        int[] result = new int[2];
        Arrays.fill(result,index);
        int left = index - 1,right = index + 1;
        int target = nums[index];
        while(left >= 0 && nums[left] == target){
            result[0] = left--;
        }
        while(right < nums.length && nums[right] == target){
            result[1] = right++;
        }
        return result;
    }
}
```

## [83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/) <a href="#83-shan-chu-pai-xu-lian-biao-zhong-de-zhong-fu-yuan-su" id="83-shan-chu-pai-xu-lian-biao-zhong-de-zhong-fu-yuan-su"></a>

```java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if(head == null || head.next == null){
            return head;
        }
        /**
        找到不同的直接pre.next即可
        */
        ListNode pre = head;
        ListNode cur = head.next;
        while(cur != null && pre != null){
            while(pre !=null && cur != null && pre.val != cur.val){
                pre = pre.next;
                cur = cur.next;
            }
            while(pre !=null && cur != null && pre.val == cur.val){
                cur = cur.next;
            }
            pre.next = cur;
        }
        return head;
    }
}
```

## [14. 最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/) <a href="#14-zui-chang-gong-gong-qian-zhui" id="14-zui-chang-gong-gong-qian-zhui"></a>

```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        if(strs.length==0)return "";
        //公共前缀比所有字符串都短，随便选一个先
        String s=strs[0];
        for (String string : strs) {
            while(!string.startsWith(s)){
                if(s.length()==0)return "";
                //公共前缀不匹配就让它变短！
                s=s.substring(0,s.length()-1);
            }
        }
        return s;
    }
}
```

## [128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/) <a href="#128-zui-chang-lian-xu-xu-lie" id="128-zui-chang-lian-xu-xu-lie"></a>

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        // 要求时间复杂度为 O(n)
        /**
        直接使用set存储，然后往后找当前item看最长多少
        */
        Set<Integer> set = new HashSet();
        for(int item:nums){
            set.add(item);
        }
        int result = 0;
        for(int i = 0; i < nums.length ; i++){
            // 排除不是第一个的
            if(!set.contains(nums[i] - 1)){
                int curLong = 1;
                int curStep = nums[i] + 1;
                while(set.contains(curStep)){
                    curStep++;
                    curLong++;
                }
                result = Math.max(curLong,result);
            }
        }
        return result;
    }
}
```

## [221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/) <a href="#221-zui-da-zheng-fang-xing" id="221-zui-da-zheng-fang-xing"></a>

```java
class Solution {
    public int maximalSquare(char[][] matrix) {
        /**
        动态规划
        dp[i][j]表示以第i行第j列为右下角所能构成的最大正方形边长
        */
        int len1 = matrix.length;
        int len2 = matrix[0].length;
        if(len1 < 1){
            return 0;
        }
        int max = 0;
        int[][] dp = new int[len1 + 1][len2 + 1];
        for(int i = 1 ;i <= len1;i++){
            for(int j = 1 ;j<= len2;j++){
                if(matrix[i-1][j-1] == '1'){
                    dp[i][j] = Math.min(dp[i-1][j-1],Math.min(dp[i-1][j],dp[i][j-1])) + 1;
                    max = Math.max(max,dp[i][j]); 
                }
            }
        }
        return max*max;
    }
}
```

## [240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/) <a href="#240-sou-suo-er-wei-ju-zhen-ii" id="240-sou-suo-er-wei-ju-zhen-ii"></a>

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        /**
        每行的元素从左到右升序排列。
        每列的元素从上到下升序排列。
        */
        if (matrix == null || matrix.length == 0) return false;
        int m = 0;
        int n = matrix[0].length - 1;
        while (m < matrix.length && n >= 0) {
            // 如果大了 X-- 如果小了 Y++
            if (matrix[m][n] == target) {
                return true;
            } else if (matrix[m][n] > target) {
                n--;
            } else {
                m++;
            }
        }
        return false;
    }
}
```

## [162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/) <a href="#162-xun-zhao-feng-zhi" id="162-xun-zhao-feng-zhi"></a>

```java
class Solution {
    public int findPeakElement(int[] nums) {
        // 必须实现时间复杂度为 O(log n) 的算法
        // 二分法
        int left = 0,right = nums.length -1;
        while(left < right){
            int mid = left + (right - left)/2;
            //规律一：如果nums[i] > nums[i+1]，则在i之前一定存在峰值元素
            //规律二：如果nums[i] < nums[i+1]，则在i+1之后一定存在峰值元素
            if(nums[mid] > nums[mid + 1]){
                right = mid;
            }else{
                left = mid + 1;
            }
        }
        return left;
    }
}
```

## [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/) <a href="#62-bu-tong-lu-jing" id="62-bu-tong-lu-jing"></a>

```java
class Solution {
    /**
    动态规划
    dp[i][j]代表的是到达该位置的所有路径总数
    */
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for(int i = 0;i < m; i++){
            for(int j = 0;j < n; j++){
                if(i == 0 || j == 0){
                    dp[i][j] = 1;
                    continue;
                }
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
}
```

## [153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/) <a href="#153-xun-zhao-xuan-zhuan-pai-xu-shu-zu-zhong-de-zui-xiao-zhi" id="153-xun-zhao-xuan-zhuan-pai-xu-shu-zu-zhong-de-zui-xiao-zhi"></a>

```java
class Solution {
    public int findMin(int[] nums) {
        // 要求O(log n)
        int left = 0,right =nums.length - 1;
        while(left < right){
            int mid = left + (right - left)/2;
            if(nums[mid] < nums[right]){
                right = mid;
            }else{
                left = mid + 1;
            }
        }
        return nums[left];
    }
}
```

## [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/) <a href="#24-liang-liang-jiao-huan-lian-biao-zhong-de-jie-dian" id="24-liang-liang-jiao-huan-lian-biao-zhong-de-jie-dian"></a>

```java
class Solution {
    public ListNode swapPairs(ListNode head) {
        if(head == null || head.next == null){
            return head;
        }
        //一共三个节点:head, next, swapPairs(next.next)
          //下面的任务便是交换这3个节点中的前两个节点
        ListNode temp = head.next;
        head.next = swapPairs(temp.next);
        temp.next = head;
        return temp;
    }
}
```

## [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/) <a href="#695-dao-yu-de-zui-da-mian-ji" id="695-dao-yu-de-zui-da-mian-ji"></a>

```java
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int max = 0;
        // 无脑dfs即可
        for(int i = 0; i < grid.length; i++){
            for(int j = 0; j < grid[0].length; j++){
                if(grid[i][j] == 1){
                    max = Math.max (dfs(grid, i, j), max);
                }
            }
        }
        return max;
    }

    public int dfs(int[][] grid, int i, int j){
        if(i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] == 0){
            return 0;
        }
        grid[i][j] = 0;
        int count = 1;
        count += dfs(grid, i+1, j);
        count += dfs(grid, i-1, j);
        count += dfs(grid, i, j+1);
        count += dfs(grid, i, j-1);
        return count;
    }
}
```

## [394. 字符串解码](https://leetcode-cn.com/problems/decode-string/) <a href="#394-zi-fu-chuan-jie-ma" id="394-zi-fu-chuan-jie-ma"></a>

```java
class Solution {
    /**
     * 双栈解法：
     * 准备两个栈，一个存放数字，一个存放字符串
     * 遍历字符串分4中情况
     * 一、如果是数字 将字符转成整型数字 注意数字不一定是个位 有可能是十位，百位等 所以digit = digit*10 + ch - '0';
     * 二、如果是字符 直接将字符放在临时字符串中
     * 三、如果是"[" 将临时数字和临时字符串入栈
     * 四、如果是"]" 将数字和字符串出栈 此时临时字符串res = 出栈字符串 + 出栈数字*res
     */
    public String decodeString(String s) {
        //创建数字栈，创建字符串栈 及临时数字和临时字符串
        Deque<Integer> stack_digit = new LinkedList<>();
        Deque<StringBuilder> stack_string = new LinkedList<>();
        int digit = 0;
        StringBuilder res = new StringBuilder();
        //遍历字符串 分4中情况
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == '[') {
                //如果是"[" 将临时数字和临时字符串入栈
                stack_digit.push(digit);
                stack_string.push(res);
                digit = 0;
                res = new StringBuilder();
            }else if (ch == ']') {
                //如果是"]" 将数字和字符串出栈 此时临时字符串res = 出栈字符串 + 出栈数字*res
                StringBuilder temp = stack_string.poll();
                int count = stack_digit.poll();
                for (int j = 0; j < count; j++) {
                    temp.append(res.toString());
                }
                res = temp;
            }else if (Character.isDigit(ch)) {
                //如果是数字 将字符转成整型数字 ch-‘0’。 注意数字不一定是个位 比如100[a] 所以digit要*10
                digit = digit*10 + ch - '0';
            }else {
                //如果是字符 直接将字符放在临时字符串中
                res.append(ch);
            }
        }
        return res.toString();
    }
}
```

## [136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/) <a href="#136-zhi-chu-xian-yi-ci-de-shu-zi" id="136-zhi-chu-xian-yi-ci-de-shu-zi"></a>

```java
class Solution {
    public int singleNumber(int[] nums) {
        /**
        所有元素的异或运算 最后只剩下单个的数
        相同的数异或为0: n ^ n => 0
        任何数于0异或为任何数 0 ^ n => n
        */
        int result = 0;
        for (int i = 0; i < nums.length; i++) {
                result = result ^ nums[i];
        }
        return result;
    }
}
```

## [468. 验证IP地址](https://leetcode-cn.com/problems/validate-ip-address/) <a href="#468-yan-zheng-ip-di-zhi" id="468-yan-zheng-ip-di-zhi"></a>

```java
class Solution {
    public String validIPAddress(String IP) {
        /**
        正则表达式
        */
        if (IP == null) {
            return "Neither";
        }

        String regex0 = "(\\d|[1-9]\\d|1\\d\\d|2[0-4]\\d|25[0-5])";
        String regexIPv4 = regex0 + "(\\." + regex0 + "){3}";
        String regex1 = "([\\da-fA-F]{1,4})";
        String regexIPv6 = regex1 + "(:" + regex1 + "){7}";

        String result = "Neither";
        if (IP.matches(regexIPv4)) {
            result = "IPv4";
        } else if (IP.matches(regexIPv6)) {
            result = "IPv6";
        }
        return result;
    }
}
```

## [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/) <a href="#122-mai-mai-gu-piao-de-zui-jia-shi-ji-ii" id="122-mai-mai-gu-piao-de-zui-jia-shi-ji-ii"></a>

```java
class Solution {
    public int maxProfit(int[] prices) {
        // 可以今天卖出今天买入 只要今天比昨天高就可以直接卖
        int result = 0;
        for(int i =1 ;i<prices.length;i++){
            int money = prices[i] - prices[i-1];
            result += money > 0?money:0;
        }
        return result;
    }
}
```

## [152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/) <a href="#152-cheng-ji-zui-da-zi-shu-zu" id="152-cheng-ji-zui-da-zi-shu-zu"></a>

```java
class Solution {
    public int maxProduct(int[] nums) {
        /**
        由于存在负数，所以需要维护两个 一个最大一个最小
        遇到负数的时候 最小的就变成最大的了
        */
        //一个保存最大的，一个保存最小的。
        int max = Integer.MIN_VALUE, imax = 1, imin = 1;
        for(int i=0; i<nums.length; i++){
            //如果数组的数是负数，那么会导致最大的变最小的，最小的变最大的。因此交换两个的值。
            if(nums[i] < 0){ 
                int tmp = imax; 
                imax = imin; 
                imin = tmp;
            } 
            imax = Math.max(imax*nums[i], nums[i]);
            imin = Math.min(imin*nums[i], nums[i]);

            max = Math.max(max, imax);
        }
        return max;
    }
}
```

## [227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/) <a href="#227-ji-ben-ji-suan-qi-ii" id="227-ji-ben-ji-suan-qi-ii"></a>

```java
class Solution {
    public int calculate(String s) {
        // 保存上一个符号，初始为 +
        char sign = '+';
        Stack<Integer> numStack = new Stack<>();
        // 保存当前数字，如：12是两个字符，需要进位累加
        int num = 0;
        int result = 0;
        for(int i = 0; i < s.length(); i++){
            char cur = s.charAt(i);
            if(cur >= '0'){
                // 记录当前数字。先减，防溢出
                num = num*10 - '0' + cur;
            }
            if((cur < '0' && cur !=' ' )|| i == s.length()-1){
                // 判断上一个符号是什么
                switch(sign){
                    // 当前符号前的数字直接压栈
                    case '+': numStack.push(num);break;
                    // 当前符号前的数字取反压栈
                    case '-': numStack.push(-num);break;
                    // 数字栈栈顶数字出栈，与当前符号前的数字相乘，结果值压栈
                    case '*': numStack.push(numStack.pop()*num);break;
                    // 数字栈栈顶数字出栈，除于当前符号前的数字，结果值压栈
                    case '/': numStack.push(numStack.pop()/num);break;
                }
                // 记录当前符号
                sign = cur;
                // 数字清零
                num = 0;
            }
        }
        // 将栈内剩余数字累加，即为结果
        while(!numStack.isEmpty()){
            result += numStack.pop();
        }
        return result;
    }
}
```

## [138. 复制带随机指针的链表](https://leetcode-cn.com/problems/copy-list-with-random-pointer/) <a href="#138-fu-zhi-dai-sui-ji-zhi-zhen-de-lian-biao" id="138-fu-zhi-dai-sui-ji-zhi-zhen-de-lian-biao"></a>

```java
class Solution {
    public Node copyRandomList(Node head) {
        /**
        在原节点后创建新节点
        然后设置新节点的随机节点
        最后分离即可
        */
        if(head==null) {
            return null;
        }
        Node p = head;
        //第一步，在每个原节点后面创建一个新节点
        //1->1'->2->2'->3->3'
        while(p!=null) {
            Node newNode = new Node(p.val);
            newNode.next = p.next;
            p.next = newNode;
            p = newNode.next;
        }
        p = head;
        //第二步，设置新节点的随机节点
        while(p!=null) {
            if(p.random!=null) {
                p.next.random = p.random.next;
            }
            p = p.next.next;
        }
        Node dummy = new Node(-1);
        p = head;
        Node cur = dummy;
        //第三步，将两个链表分离
        while(p!=null) {
            cur.next = p.next;
            cur = cur.next;
            p.next = cur.next;
            p = p.next;
        }
        return dummy.next;
    }
}
```

## [179. 最大数](https://leetcode-cn.com/problems/largest-number/) <a href="#179-zui-da-shu" id="179-zui-da-shu"></a>

```java
class Solution {
    public String largestNumber(int[] nums) {
        int n = nums.length;
        String numsToWord[] = new String[n];
        for(int i=0;i<n;i++){
            numsToWord[i] = String.valueOf(nums[i]);
        }
        //compareTo()方法比较的时候是按照ASCII码逐位比较的
        //通过比较(a+b)和(b+a)的大小，就可以判断出a,b两个字符串谁应该在前面
        //所以[3,30,34]排序后变为[34,3,30]
        //[233，23333]排序后变为[23333，233]
        Arrays.sort(numsToWord,(a,b)->{
            return (b+a).compareTo(a+b);
        });
        //如果排序后的第一个元素是0，那后面的元素肯定小于或等于0，则可直接返回0
        if(numsToWord[0].equals("0")){
            return "0";
        }
        StringBuilder sb = new StringBuilder();
        for(int i=0;i<n;i++){
            sb.append(numsToWord[i]);
        }
        return sb.toString();
    }
}
```

## [662. 二叉树最大宽度](https://leetcode-cn.com/problems/maximum-width-of-binary-tree/) <a href="#662-er-cha-shu-zui-da-kuan-du" id="662-er-cha-shu-zui-da-kuan-du"></a>

```java
class Solution {

    private int maxW = 0;

    public int widthOfBinaryTree(TreeNode root) {
        /**
        假设满二叉树表示成数组序列, 根节点所在的位置为1, 则任意位于i节点的左右子节点的index为2*i, 2*i+1
        用一个List保存每层的左端点, 易知二叉树有多少层List的元素就有多少个. 那么可以在dfs的过程中记录每个
        节点的index及其所在的层level, 如果level > List.size()说明当前节点就是新的一层的最左节点, 将其
        加入List中, 否则判断当前节点的index减去List中对应层的最左节点的index的宽度是否大于最大宽度并更新
        **/
        dfs(root, 1, 1, new ArrayList<>());
        return maxW;
    }

    private void dfs(TreeNode r, int level, int index, List<Integer> left) {
        if(r == null) return;
        if(level > left.size()) left.add(index);
        maxW = Math.max(maxW, index - left.get(level-1) + 1);
        dfs(r.left, level+1, index*2, left);
        dfs(r.right, level+1, index*2+1, left);
    }
}
```

## [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/) <a href="#209-chang-du-zui-xiao-de-zi-shu-zu" id="209-chang-du-zui-xiao-de-zi-shu-zu"></a>

```java
class Solution {

    // 滑动窗口
    public int minSubArrayLen(int s, int[] nums) {
        int left = 0;
        int sum = 0;
        int result = Integer.MAX_VALUE;
        for (int right = 0; right < nums.length; right++) {
            sum += nums[right];
            while (sum >= s) {
                result = Math.min(result, right - left + 1);
                sum -= nums[left++];
            }
        }
        return result == Integer.MAX_VALUE ? 0 : result;
    }
}
```

## [283. 移动零](https://leetcode-cn.com/problems/move-zeroes/) <a href="#283-yi-dong-ling" id="283-yi-dong-ling"></a>

```java
class Solution {
    // 思路：设置一个index，表示非0数的个数，循环遍历数组，
    // 如果不是0，将非0值移动到第index位置,然后index + 1
    //遍历结束之后，index值表示为非0的个数，再次遍历，从index位置后的位置此时都应该为0
    public void moveZeroes(int[] nums) {
        if (nums == null || nums.length <= 1) {
            return;
        }
        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[index] = nums[i];
                index++;
            }
        }

        for (int i = index; i < nums.length; i++) {
            nums[i] = 0;
        }
    }
}
```

## [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/) <a href="#jian-zhi-offer36-er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao" id="jian-zhi-offer36-er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao"></a>

```java
class Solution {
    // 1. 中序，递归，来自解题大佬
    Node pre, head;
    public Node treeToDoublyList(Node root) {
        // 边界值
        if(root == null) return null;
        dfs(root);

        // 题目要求头尾连接
        head.left = pre;
        pre.right = head;
        // 返回头节点
        return head;
    }
    void dfs(Node cur) {
        // 递归结束条件
        if(cur == null) return;
        dfs(cur.left);
        // 如果pre为空，就说明是第一个节点，头结点，然后用head保存头结点，用于之后的返回
        if (pre == null) head = cur;
        // 如果不为空，那就说明是中间的节点。并且pre保存的是上一个节点，
        // 让上一个节点的右指针指向当前节点
        else if (pre != null) pre.right = cur;
        // 再让当前节点的左指针指向父节点，也就连成了双向链表
        cur.left = pre;
        // 保存当前节点，用于下层递归创建
        pre = cur;
        dfs(cur.right);
    }
}
```

## [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/) <a href="#198-da-jia-jie-she" id="198-da-jia-jie-she"></a>

```java
class Solution {
    public int rob(int[] nums) {
        // 动态规划
        // dp[i] = Math.max(dp[i-1],dp[i-2] + nums[i]);
        if(nums.length == 0){
            return 0;
        }else if(nums.length == 1){
            return nums[0];
        }
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0],nums[1]);
        for(int i = 2; i < nums.length ; i++){
            dp[i] = Math.max(dp[i-1],dp[i-2] + nums[i]);
        }
        return dp[nums.length - 1];
    }
}
```

## [297. 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/) <a href="#297-er-cha-shu-de-xu-lie-hua-yu-fan-xu-lie-hua" id="297-er-cha-shu-de-xu-lie-hua-yu-fan-xu-lie-hua"></a>

```java
public class Codec {
    /**
    DFS + Stringbuilder
    */

    public String serialize(TreeNode root) {      //用StringBuilder
        StringBuilder res = ser_help(root, new StringBuilder());
        return res.toString();
    }

    public StringBuilder ser_help(TreeNode root, StringBuilder str){
        if(null == root){
            str.append("null,");
            return str;
        }
        str.append(root.val); 
        str.append(",");
        str = ser_help(root.left, str);
        str = ser_help(root.right, str);
        return str;
    }

    public TreeNode deserialize(String data) {
        String[] str_word = data.split(",");
        List<String> list_word = new LinkedList<String>(Arrays.asList(str_word));
        return deser_help(list_word);
    }

    public TreeNode deser_help(List<String> li){
        if(li.get(0).equals("null")){
            li.remove(0);
            return null;
        }
        TreeNode res = new TreeNode(Integer.valueOf(li.get(0)));
        li.remove(0);
        res.left = deser_help(li);
        res.right = deser_help(li);
        return res;
    }
}
```

## [剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/) <a href="#jian-zhi-offer09-yong-liang-ge-zhan-shi-xian-dui-lie" id="jian-zhi-offer09-yong-liang-ge-zhan-shi-xian-dui-lie"></a>

```java
class CQueue {
    /**
    维护两个栈
    输入栈inputStack应用存放
    输出栈outputStack是每当需要弹出的时候将inputStack的转换到outputStack中
    */
    LinkedList<Integer> inputStack;
    LinkedList<Integer> outputStack;

    public CQueue() {
        inputStack = new LinkedList<>();
        outputStack = new LinkedList<>();
    }

    public void appendTail(int value) {
        inputStack.add(value);
    }

    public int deleteHead() {
        convert();
        return outputStack.isEmpty()?-1:outputStack.pop();
    }

    private void convert(){
        if (outputStack.isEmpty()) {
            if (inputStack.isEmpty()) return;
            while (!inputStack.isEmpty()) {
                outputStack.add(inputStack.pop());
            }
        }
    }
}
```

## [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/) <a href="#jian-zhi-offer54-er-cha-sou-suo-shu-de-dikda-jie-dian" id="jian-zhi-offer54-er-cha-sou-suo-shu-de-dikda-jie-dian"></a>

```java
class Solution {
    /**
    二叉搜索树的一个特性：通过中序遍历所得到的序列，就是有序的。
    */
    public int kthLargest(TreeNode root, int k) {
        // clarification:  root == null?   k <= 1?
        List<Integer> list = new ArrayList<>();
        helper(root, list);
        return list.get(list.size() - k);
    }

    private void helper(TreeNode root, List<Integer> list) {
        if (root == null) return;
        if (root.left != null) helper(root.left, list);
        list.add(root.val);
        if (root.right != null) helper(root.right, list);
    }
}
```

## [958. 二叉树的完全性检验](https://leetcode-cn.com/problems/check-completeness-of-a-binary-tree/) <a href="#958-er-cha-shu-de-wan-quan-xing-jian-yan" id="958-er-cha-shu-de-wan-quan-xing-jian-yan"></a>

```java
class Solution {
    /**
    bfs 当出现 null 值时停止遍历，
    如果此时还有没有遍历到的结点，说明该树非完全二叉树。
    */
    public boolean isCompleteTree(TreeNode root) {
        if(root == null){
            return false;
        }
        Deque<TreeNode> deque = new LinkedList<>();
        deque.offerLast(root);
        TreeNode cur;
        while((cur = deque.pollFirst())!= null){
            deque.offerLast(cur.left);
            deque.offerLast(cur.right);
        }
        while(!deque.isEmpty()){
            if(deque.pollFirst()!=null){
                return false;
            }
        }
        return true;
    }
}
```

## [498. 对角线遍历](https://leetcode-cn.com/problems/diagonal-traverse/) <a href="#498-dui-jiao-xian-bian-li" id="498-dui-jiao-xian-bian-li"></a>

```java
class Solution {
    public int[] findDiagonalOrder(int[][] matrix) {
    // 遍历方向由层数决定，而层数即为横纵坐标之和。
    if (matrix == null || matrix.length == 0) {
        return new int[]{};
    }
    int r = 0, c = 0;
    int row = matrix.length, col = matrix[0].length;
    int[] res = new int[row * col];
    for (int i = 0; i < res.length; i++) {
        res[i] = matrix[r][c];
        // r + c 即为遍历的层数，偶数向上遍历，奇数向下遍历
        if ((r + c) % 2 == 0) {
            if (c == col - 1) {
                // 往下移动一格准备向下遍历
                r++;
            } else if (r == 0) {
                // 往右移动一格准备向下遍历
                c++;
            } else {
                // 往上移动
                r--;
                c++;
            }
        } else {
            if (r == row - 1) {
                // 往右移动一格准备向上遍历
                c++;
            } else if (c == 0) {
                // 往下移动一格准备向上遍历
                r++;
            } else {
                // 往下移动
                r++;
                c--;
            }
        }
    }
    return res;
}
}
```

## [402. 移掉 K 位数字](https://leetcode-cn.com/problems/remove-k-digits/) <a href="#402-yi-diaokwei-shu-zi" id="402-yi-diaokwei-shu-zi"></a>

```java
class Solution {
    public String removeKdigits(String num, int k) {
        /**
        用一个栈维护当前的答案序列，栈中的元素代表截止到当前位置，删除不超过 
        k次个数字后，所能得到的最小整数。根据之前的讨论：在使用k个删除次数之前
        ,栈中的序列从栈底到栈顶单调不降。
        */
        Deque<Character> deque = new LinkedList<Character>();
        int length = num.length();
        for (int i = 0; i < length; ++i) {
            char digit = num.charAt(i);
            while (!deque.isEmpty() && k > 0 && deque.peekLast() > digit) {
                deque.pollLast();
                k--;
            }
            deque.offerLast(digit);
        }

        for (int i = 0; i < k; ++i) {
            deque.pollLast();
        }

        StringBuilder ret = new StringBuilder();
        boolean leadingZero = true;
        while (!deque.isEmpty()) {
            char digit = deque.pollFirst();
            if (leadingZero && digit == '0') {
                continue;
            }
            leadingZero = false;
            ret.append(digit);
        }
        return ret.length() == 0 ? "0" : ret.toString();
    }
}
```

## [145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/) <a href="#145-er-cha-shu-de-hou-xu-bian-li" id="145-er-cha-shu-de-hou-xu-bian-li"></a>

```java
// // 前序遍历顺序：中-左-右，入栈顺序：中-右-左
// class Solution {
//     public List<Integer> preorderTraversal(TreeNode root) {
//         List<Integer> result = new ArrayList<>();
//         if (root == null){
//             return result;
//         }
//         Stack<TreeNode> stack = new Stack<>();
//         stack.push(root);
//         while (!stack.isEmpty()){
//             TreeNode node = stack.pop();
//             result.add(node.val);
//             if (node.right != null){
//                 stack.push(node.right);
//             }
//             if (node.left != null){
//                 stack.push(node.left);
//             }
//         }
//         return result;
//     }
// }

// // 中序遍历顺序: 左-中-右 入栈顺序： 左-右
// class Solution {
//     public List<Integer> inorderTraversal(TreeNode root) {
//         List<Integer> result = new ArrayList<>();
//         if (root == null){
//             return result;
//         }
//         Stack<TreeNode> stack = new Stack<>();
//         TreeNode cur = root;
//         while (cur != null || !stack.isEmpty()){
//            if (cur != null){
//                stack.push(cur);
//                cur = cur.left;
//            }else{
//                cur = stack.pop();
//                result.add(cur.val);
//                cur = cur.right;
//            }
//         }
//         return result;
//     }
// }

// 后序遍历顺序 左-右-中 入栈顺序：中-左-右 出栈顺序：中-右-左， 最后翻转结果
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null){
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()){
            TreeNode node = stack.pop();
            result.add(node.val);
            if (node.left != null){
                stack.push(node.left);
            }
            if (node.right != null){
                stack.push(node.right);
            }
        }
        Collections.reverse(result);
        return result;
    }
}
```

## [560. 和为 K 的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/) <a href="#560-he-weikde-zi-shu-zu" id="560-he-weikde-zi-shu-zu"></a>

```java
class Solution {
    /**
    遍历数组nums，计算从第0个元素到当前元素的和，
    用哈希表保存出现过的累积和sum的次数。如果sum - k在哈希表中出现过，
    则代表从当前下标i往前有连续的子数组的和为sum。
    */
    public int subarraySum(int[] nums, int k) {
        // 利用哈希表实现线性寻找
        HashMap <Integer, Integer> hashMap = new HashMap<>();
        // 和为0的连续子数组出现1次。
        hashMap.put(0, 1);
        int i;
        int sum = 0;
        int count = 0;
        for (i = 0; i < nums.length; i++)
        {
            sum += nums[i];
            if (hashMap.containsKey(sum - k))   // 表示连续子数组减去连续子数组，必定为连续子数组
            {
                count += hashMap.get(sum - k);
            }
            hashMap.put(sum, hashMap.getOrDefault(sum, 0) + 1);
        }
        return count;
    }
}
```
