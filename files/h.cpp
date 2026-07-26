class Solution {
public:
    int maxSubArray(vector<int>& arr) {
       int res = arr[0], curr = arr[0];
        for (int i = 1; i < arr.size(); i++) {
            curr = max(arr[i], curr + arr[i]);
            res = max(res, curr);
        }
        return res; 
    }
};