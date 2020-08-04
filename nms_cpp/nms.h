#include <vector>
#include <string>
#include <algorithm>

#include "polyiou.h"

using namespace std;

struct BBox
{
	float iou(const BBox &, bool rect)const;
	string cla;
	float score;
	float x1, y1, x2, y2, x3, y3, x4, y4;
};

float BBox::iou(const BBox &bbox, bool rect)const
{
	if (rect)
	{
		float xmin = bbox.x1, ymin = bbox.y1, xmax = bbox.x3, ymax = bbox.y3;
		float x_lt = max(x1, xmin);
		float y_lt = max(y1, ymin);
		float x_rb = min(x3, xmax);
		float y_rb = min(y3, ymax);
		float inter_area = max(x_rb - x_lt, static_cast<float>(0.))*max(y_rb - y_lt, static_cast<float>(0.));
		float union_area = (xmax - xmin)*(ymax - ymin) + (x3 - x1)*(y3 - y1);
		return inter_area / (union_area - inter_area);
	}
	else
		return iou_poly({ x1, y1, x2, y2, x3, y3, x4, y4 },
		{ bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.x3, bbox.y3, bbox.x4, bbox.y4 });
}

struct ListNode
{
	ListNode(const BBox &bbox, ListNode *next = nullptr) :val(bbox), next(next) {}
	BBox val;
	ListNode *next;
};

vector<BBox> nms(vector<BBox> &bboxes, const vector<string> &cls, float thresh, bool rect) //rect:Õý¿òÎªtrue, Ð±¿òÎªfalse
{
	sort(bboxes.begin(), bboxes.end(), [](const BBox &bb1, const BBox &bb2) {return bb1.score > bb2.score; });
	vector<BBox>ret;
	for (auto c : cls)
	{
		ListNode *head = new ListNode(BBox());
		ListNode *cur = head;
		for (auto i : bboxes)
			if (c == i.cla)
			{
				cur->next = new ListNode(i);
				cur = cur->next;
			}

		while (head->next)
		{
			ListNode *front = head->next;
			ret.push_back(front->val);
			head->next = head->next->next;
			if (!head->next)
			{
				delete front;
				break;
			}
			ListNode *pre = head;
			cur = head->next;
			while (cur)
			{
				float iou = front->val.iou(cur->val, rect);
				if (iou > thresh)
				{
					pre->next = cur->next;
					delete cur;
					cur = pre->next;
				}
				else
				{
					pre = pre->next;
					cur = cur->next;
				}
			}
			delete front;
		}
		delete head;
	}
	return ret;
}
