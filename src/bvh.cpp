#pragma once
#include <numeric>
#include <cassert>
#include "scene.h"

static int tnodeNum;

TBB::TBB() :_min(glm::vec3(FLT_MAX)), _max(glm::vec3(FLT_MIN)) {}
TBB::TBB(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2) :_min(glm::min(v0, glm::min(v1, v2))), _max(glm::max(v0, glm::max(v1, v2))) {}
TBB::TBB(glm::vec3 min, glm::vec3 max) :_min(min), _max(max) {}

void TBB::expand(const glm::vec3& p)
{
    _max = glm::max(_max, p);
    _min = glm::min(_min, p);
}

void TBB::expand(const TBB& other)
{
    expand(other._max);
    expand(other._min);
}

void sortAxis(std::vector<TriangleDetail>& triangles, std::vector<int>& objIdx, int axis)
{
    std::sort(objIdx.begin(), objIdx.end(), [axis, triangles](const int& a, const int& b) {return triangles[a].centroid[axis] < triangles[b].centroid[axis]; });
}


int splitBVH(std::vector<TriangleDetail>& triangles, std::vector<int> objIdx, int num, TBB& tbb, int face, std::vector<std::vector<TBVHNode>>& mnode)
{
    std::vector<int>leftIndex, rightIndex;
    if (num <= 1) {
        int id = tnodeNum;
        tnodeNum++;
        mnode[face][id].isLeaf = true;
        mnode[face][id].tbb = tbb;
        if (num == 0) {
            mnode[face][id].triId = -1;
        }
        else {
            mnode[face][id].triId = objIdx[0];
        }
        return id;
    }
    int axis = 0;
    int index = 0;
    float bestCost = FLT_MAX;
    TBB bestTBBLeft, bestTBBRight;

    if (num == 2) {
        leftIndex = { objIdx[0] };
        rightIndex = { objIdx[1] };
        bestTBBLeft = triangles[objIdx[0]].tbb;
        bestTBBRight = triangles[objIdx[1]].tbb;
    }
    else {
        std::vector<int> bestObjIdx;
        for (size_t i = 0; i < 3; i++)
        {
            sortAxis(triangles, objIdx, i);
            std::vector<float> leftArea(num);
            std::vector<TBB>leftTBBs(num);
            float cost = 0.f;
            TBB tmpTBB;
            for (size_t j = 0; j < num; j++)
            {
                tmpTBB.expand(triangles[objIdx[j]].tbb);
                leftArea[j] = tmpTBB.area();
                leftTBBs[j] = tmpTBB;
            }
            tmpTBB = TBB();
            for (size_t j = num - 1; j > 0; j--)
            {
                tmpTBB.expand(triangles[objIdx[j]].tbb);
                const float tempCost = j * leftArea[j] + (num - j) * tmpTBB.area();
                if (tempCost < bestCost) {
                    index = j - 1;
                    bestCost = tempCost;
                    axis = i;
                    bestTBBLeft = leftTBBs[index];
                    bestTBBRight = tmpTBB;
                }
            }
            if (axis == i) {
                bestObjIdx = objIdx;
            }
        }
        leftIndex.assign(bestObjIdx.begin(), bestObjIdx.begin() + index + 1);
        rightIndex.assign(bestObjIdx.begin() + index + 1, bestObjIdx.end());
    }

    int id = tnodeNum;
    tnodeNum++;
    mnode[face][id].tbb = tbb;
    mnode[face][id].isLeaf = false;

    int idLeft = splitBVH(triangles, leftIndex, index + 1, bestTBBLeft, face, mnode);
    int idRight = splitBVH(triangles, rightIndex, num - index - 1, bestTBBRight, face, mnode);
    if (bestTBBLeft._min.x > bestTBBRight._min.x)
        std::swap(idLeft, idRight);
    mnode[face][id].left = idLeft;
    mnode[face][id].right = idRight;
    return id;
}



void ReorderNodes(std::vector<TriangleDetail>& triangles, int face, int index, std::vector<std::vector<TBVHNode>>  mnode)
{
    if (index < 0) return;
    if ((unsigned int)tnodeNum == (triangles.size() * 2)) return;

    tnodeNum++;
    int temp_id = tnodeNum - 1;
    mnode[face][temp_id] = mnode[6][index];
    mnode[face][temp_id].base = index;

    if (mnode[6][index].isLeaf) return;

    ReorderNodes(triangles, face, mnode[6][index].left, mnode);
    ReorderNodes(triangles, face, mnode[6][index].right, mnode);
}


int ReorderTree(std::vector<TriangleDetail>& triangles, int face, int index, std::vector<std::vector<TBVHNode>>  mnode)
{
    if (mnode[6][index].isLeaf)
    {
        tnodeNum++;
        return tnodeNum - 1;
    }

    tnodeNum++;
    int temp_id = tnodeNum - 1;
    mnode[face][temp_id].left = ReorderTree(triangles, face, mnode[6][index].left, mnode);
    mnode[face][temp_id].right = ReorderTree(triangles, face, mnode[6][index].right, mnode);
    return temp_id;
}


void SetLeftMissLinks(int id, int idParent, int face, std::vector<std::vector<TBVHNode>>  mnode)
{
    if (mnode[face][id].isLeaf)
    {
        mnode[face][id].miss = id + 1;
        return;
    }

    mnode[face][id].miss = mnode[face][idParent].right;

    SetLeftMissLinks(mnode[face][id].left, id, face, mnode);
    SetLeftMissLinks(mnode[face][id].right, id, face, mnode);
}


void SetRightMissLinks(int id, int idParent, int face, std::vector<std::vector<TBVHNode>>  mnode)
{
    if (mnode[face][id].isLeaf)
    {
        mnode[face][id].miss = id + 1;
        return;
    }

    if (mnode[face][idParent].right == id)
    {
        mnode[face][id].miss = mnode[face][idParent].miss;
    }

    SetRightMissLinks(mnode[face][id].left, id, face, mnode);
    SetRightMissLinks(mnode[face][id].right, id, face, mnode);
}


TBVH::TBVH(std::vector<TriangleDetail>& triangles, TBB& tbb) :nodes(std::vector<std::vector<TBVHNode>>(7, std::vector<TBVHNode>(triangles.size() * 2)))
{
    for (int face = 0; face <= 5; face++)
    {
        const int num = triangles.size();
        std::vector<int>objIdx(num);
        std::iota(objIdx.begin(), objIdx.end(), 0);
        tnodeNum = 0;
        for (int i = 0; i < num * 2; i++) {
            nodes[face][i].miss = -1;
            nodes[face][i].base = i;
        }

        if (face == 0)
        {
            splitBVH(triangles, objIdx, num, tbb, 0, nodes);
            this->nodesNum = tnodeNum;

            for (int i = 0; i <= num * 2 - 1; i++)
            {
                nodes[6][i].miss = -1;
            }
        }
        else
        {
            for (int i = 0; i <= this->nodesNum - 1; i++)
            {
                nodes[6][i] = nodes[0][i];
            }

            for (int i = 0; i <= this->nodesNum - 1; i++)
            {
                if (nodes[6][i].isLeaf) continue;

                if ((face == 1) && (nodes[6][nodes[6][i].left].tbb._max.x > nodes[6][nodes[6][i].right].tbb._max.x)) continue;
                if ((face == 2) && (nodes[6][nodes[6][i].left].tbb._min.y < nodes[6][nodes[6][i].right].tbb._min.y)) continue;
                if ((face == 3) && (nodes[6][nodes[6][i].left].tbb._max.y > nodes[6][nodes[6][i].right].tbb._max.y)) continue;
                if ((face == 4) && (nodes[6][nodes[6][i].left].tbb._min.z < nodes[6][nodes[6][i].right].tbb._min.z)) continue;
                if ((face == 5) && (nodes[6][nodes[6][i].left].tbb._max.z > nodes[6][nodes[6][i].right].tbb._max.z)) continue;

                const int temp = nodes[6][i].left;
                nodes[6][i].left = nodes[6][i].right;
                nodes[6][i].right = temp;
            }

            tnodeNum = 0;
            ReorderNodes(triangles, face, 0, nodes);
            tnodeNum = 0;
            ReorderTree(triangles, face, 0, nodes);
        }

        nodes[face][0].miss = -1;
        SetLeftMissLinks(0, 0, face, nodes);
        nodes[face][0].miss = -1;
        SetRightMissLinks(0, 0, face, nodes);
        nodes[face][0].miss = -1;
    }
}
