# 刷题

## 简介

下面的题目按照[https://codetop.cc/home](https://codetop.cc/home)出现的频率从高到低排序。

## [206. 反转链表 - 力扣（LeetCode）](https://leetcode-cn.com/problems/reverse-linked-list/)

```
class Solution {
    public ListNode reverseList(ListNode head) {
        /**
        迭代
        每次循环将当前cur的next指向pre
        然后pre和cur依次后移一位
         */
        ListNode pre = null;
        while(head!=null){
            ListNode temp = head.next;
            head.next = pre ;
            pre = head;
            head = temp;
        }
        return pre ;
        //return recurse(null,head);
    }

    /**
    递归
    先想好跳出递归的条件，无非是链表遍历完了
    然后是本次递归需要做的事情，首先是更新当前指针
    然后将当前指针的next指向pre，然后继续下一次递归即可
     */
    public ListNode recurse(ListNode result,ListNode head){
        if(head==null){
            return result;
        }
        ListNode temp = head.next;
        head.next = result;
        return recurse(head,temp);
    }
}
```

## [146. LRU 缓存 - 力扣（LeetCode）](https://leetcode-cn.com/problems/lru-cache/submissions/)

```
/**
思路
使用哈希表+双向循环链表
有前继节点以及后继节点，删除与增加的时候都是O1的时间复杂度
哈希表实现获取的时候O1时间复杂度
需要注意的就是put的时候判断下，是否超过缓存
get、put的时候都需要更新缓存列表，使最新使用的节点在head
*/
public class LRUCache {
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

## [3. 无重复字符的最长子串 - 力扣（LeetCode）](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/submissions/)

```
import java.util.*;
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int[] intArray = new int[256];
        for(int i=0;i<256;i++){
            intArray[i]=0;
        }
        /**
        思路：维护一个动态窗口
        基本就是一个套路的方式
        主要是填充窗口右滑和左滑区域的数据更新
        */
        int result = 0;
        int left=0,right = 0;
        char[] array = s.toCharArray();
        while(right < array.length){
            char c = array[right];
            // 窗口右滑
            right++;
            // 窗口右滑 更新窗口区域数据
            intArray[c]++;
            // 窗口需要左滑的条件
            while(intArray[c]>=2){
                char d = array[left];
                // 窗口左滑
                left++;
                // 窗口左滑 更新窗口区域数据
                intArray[d]--;
            }
            result = Math.max(result,right-left);
        }
        return result;
    }
}
```

## [215. 数组中的第K个最大元素 - 力扣（LeetCode）](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/submissions/)

```
class Solution {
    public int findKthLargest(int[] nums, int k) {
        /**
        这里直接使用快速排序，其实可以继续优化
        比如当排序到n-k的时候直接停止，返回k坐标的value
        */
        QuickSort(nums,0,nums.length-1);
        return nums[nums.length-k];
    }

    public void QuickSort(int[] array,int start,int end){
        //出口条件 不需要再排了
        if(start>=end) return;
        int left=start,right=end;
        //必须随机
        int temp=start+(int)Math.random()*(end-start+1);
        //如果不相等 重复做逼近操作
        while(left<right){
            //如果大的话 直接略过
            while(right>left&&array[right]>=array[temp]){
                right--;
            }
            //此时跳出while之后一定是小于等于temp索引的值的
            while(right>left&&array[left]<=array[temp]){
                left++;
            }
            //此时将找到的值交换
            swap(array,left,right);
        }
        //left和right相等 则直接交换temp的值和left的值即可 也算是找到了temp所应该在的位置
        swap(array,temp,left);
        //对基准值左边的元素进行排序
        QuickSort(array,start,left-1);
        //对基准值右边的元素进行排序
        QuickSort(array,right+1,end);
    }

    public void swap(int[] array,int i,int j){
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}
```

## [25. K 个一组翻转链表 - 力扣（LeetCode）](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/submissions/)

```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        if(head==null){
            return null;
        }
        ListNode left = head,right = head;
        /**
        数k个，如果不满足k个不用反转直接返回head
         */
        for(int i=0;i<k;i++){
            if(right==null){
                return head;
            }
            right = right.next;
        }
        ListNode temp = reverse(left,right);
        left.next = reverseKGroup(right,k);
        return temp;
    }
 
    /**
    翻转从left到right的链表
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

## [15. 三数之和 - 力扣（LeetCode）](https://leetcode-cn.com/problems/3sum/submissions/)

```
class Solution {
    /**
    思路
    首先将三元组转换为二元组，记得剔除重复数据（排除第一个重复的就行）
    二元组中用排好序的数据，去获取他们的和，同时也需要排除数据，第一个不重复即可
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
            // 关键在于剔除重复数据，不让第一个数重复即可
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
            int sum = list[left]+list[right];
            int curL = list[left],curR = list[right];
            if(sum<target){
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

## [912. 排序数组 - 力扣（LeetCode）](https://leetcode-cn.com/problems/sort-an-array/submissions/)

```
import java.util.*;

class Solution {
    public int[] sortArray(int[] nums) {
        quickSort(nums,0,nums.length-1);
        return nums;
    }

    void quickSort(int[] nums, int start, int end) {
        if (start >= end) return;
        int left = start;
        int right = end;
        // 1
        // 选取随机值，防止基本有序的数组时间复杂度由O(nlog2n)退化成O(n^2)
        swap(nums, start, (start + end) / 2);
        int target = nums[start];
        while (left < right) {
            // 先找小于基准值的元素，方便跳出循环时调整基准值的位置
            while (left < right && nums[right] > target) right--;
            // 再找大于等于基准值的元素（条件有等于是为了将基准值与刚才小于基准值的元素对调）
            while (left < right && nums[left] <= target) left++;
            // 2
            if (left != right) swap(nums, left, right);
        }
        // 3
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

## [53. 最大子数组和 - 力扣（LeetCode）](https://leetcode-cn.com/problems/maximum-subarray/submissions/)

```
class Solution {
    public int maxSubArray(int[] nums) {
        //dp[target]代表下标为target的之前连续子数组最大和
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

## [1. 两数之和 - 力扣 （LeetCode）](https://leetcode-cn.com/problems/two-sum/)

```
import java.util.*;
class Solution {
    public int[] twoSum(int[] nums, int target) {
        List<Integer> list = new ArrayList();
        for(int i=0;i<nums.length;i++){
            list.add(nums[i]);
        }
        // 一定要先做排序
        Arrays.sort(nums);
        int left = 0,right = nums.length -1;
        int[] result = new int[2];
        while(left<right){
            int sum = nums[left] + nums[right];
            int start = nums[left],end = nums[right];
            if(sum==target){
                result[0] = list.indexOf(start);
                result[1] = list.lastIndexOf(end);
                break;
            }else if(sum>target){
                // 排除重复项
                while(left<right&&nums[right]==end){
                    right--;
                }
            }else {
                while(left<right&&nums[left]==start){
                    left++;
                }
            }
        }
        return result;
    }
}
```

## [21. 合并两个有序链表 - 力扣（LeetCode）](https://leetcode-cn.com/problems/merge-two-sorted-lists/submissions/)

```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
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

## [141. 环形链表 - 力扣（LeetCode）](https://leetcode-cn.com/problems/linked-list-cycle/)

```
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    /**
    快慢指针 最终相遇代表有环
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

## [102. 二叉树的层序遍历 - 力扣（LeetCode）](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
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

## [121. 买卖股票的最佳时机 - 力扣（LeetCode）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/submissions/)

```
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

## [160. 相交链表 - 力扣（LeetCode）](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
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

## [103. 二叉树的锯齿形层序遍历 - 力扣（LeetCode）](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/submissions/)

```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
        public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
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

## [20. 有效的括号 - 力扣（LeetCode）](https://leetcode-cn.com/problems/valid-parentheses/submissions/)

```
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
                // 注意是else if
            } else if(stack.empty()||stack.pop()!=c){
                return false;
            }
        }
        return stack.empty()?true:false;
    }
}
```

## [236. 二叉树的最近公共祖先 - 力扣（LeetCode）](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/submissions/)

```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
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

## [88. 合并两个有序数组 - 力扣（LeetCode）](https://leetcode-cn.com/problems/merge-sorted-array/submissions/)

```
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

## [33. 搜索旋转排序数组 - 力扣（LeetCode）](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/submissions/)

```
class Solution {
    public int search(int[] nums, int target) {
        /**
        如果中间的数小于最右边的数，则右半段是有序的，若中间数大于最右边数，则左半段是有序的，
        我们只要在有序的半段里用首尾两个数组来判断目标值是否在这一区域内，这样就可以确定保留哪半边了

        这里选择一个简单的A了，上述思路是logn的时间复杂度 更优
         */
        for(int i=0;i<nums.length;i++){
            if(nums[i]==target){
                return i;
            }
        }
        return -1;
    }
}
```

## [5. 最长回文子串 - 力扣（LeetCode）](https://leetcode-cn.com/problems/longest-palindromic-substring/)

```
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

## [200. 岛屿数量 - 力扣（LeetCode）](https://leetcode-cn.com/problems/number-of-islands/submissions/)

```
class Solution {
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

## [46. 全排列 - 力扣（LeetCode）](https://leetcode-cn.com/problems/permutations/submissions/)

```
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

## [415. 字符串相加 - 力扣（LeetCode）](https://leetcode-cn.com/problems/add-strings/submissions/)

```
class Solution {
    public String addStrings(String num1, String num2) {
        /**
        题目中明确说明，不能使用BigInteger以及转换为整数形式
        那么可以直接手动金威，每次使用%取得余数，添加，最后翻转即可
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

## [92. 反转链表 II - 力扣（LeetCode）](https://leetcode-cn.com/problems/reverse-linked-list-ii/submissions/)

```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
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

## [23. 合并K个升序链表 - 力扣（LeetCode）](https://leetcode-cn.com/problems/merge-k-sorted-lists/submissions/)

```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
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

## [142. 环形链表 II - 力扣（LeetCode）](https://leetcode-cn.com/problems/linked-list-cycle-ii/submissions/)

```
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
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

## [54. 螺旋矩阵 - 力扣（LeetCode）](https://leetcode-cn.com/problems/spiral-matrix/submissions/)

```
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        /**
        通过遍历完成一行或一列之后重新设定边界来排除重复
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

\
\
