#ifndef SHTENSOR_BTREE_H
#define SHTENSOR_BTREE_H

#include "Definitions.h"
#include "Logger.h"
#include "Utils.h"

#include <functional>
#include <list>

namespace Shtensor
{

// A Btree that is constructed within a fixed-size memory region ???
template <typename T>
class BTree
{
 public:

  struct Node
  {
    int order;
    int size;

    uint8_t data;

    static int64_t data_size(int _order)
    {
      const int ssize = 
        2 * SSIZEOF(int) // struct member vars
        + _order*SSIZEOF(Node*) // pointers to children
        + (_order-1)*SSIZEOF(int64_t) // indices
        + (_order-1)*SSIZEOF(T) // values
      ;

      // round to next multiple of 16 for correct alignment
      return Utils::round_next_multiple(ssize, 16);
    } 

    Node** children()
    {
      return reinterpret_cast<Node**>(&data);
    }

    Node** children_end()
    {
      return children() + size + 1;
    }

    int64_t* indices()
    {
      return reinterpret_cast<int64_t*>(&data + order*SSIZEOF(Node*));
    }

    int64_t* indices_end()
    {
      return indices() + size;
    }

    T* values()
    {
      return reinterpret_cast<T*>(&data + order*SSIZEOF(Node*) + (order-1)*SSIZEOF(int64_t));
    }

    T* values_end()
    {
      return values() + size;
    }

    Node(int _order)
      : order(_order)
      , size(0)
    {
    }

  };

  BTree(int order);

  ~BTree();

  void insert(int64_t _index, T _value);

  void remove(int64_t _index);

  void insert_node(Node* _node, std::vector<Node*>& _path, int _depth, int64_t _index, T _value, Node* _p_neighbor);

  void remove_node(Node* _node, std::vector<Node*>& _path, int _depth, int64_t _index);

  Node* get_neighbour(Node* _node, std::vector<Node*>& _path, int _depth, bool _left);

  Node* allocate_node();

  void deallocate_node(Node* _pnode);

  std::string make_pretty();
 
 private:

  void node_depth_init(std::vector<Node*>& _path, int64_t& _size)
  {
    _path = std::vector<Node*>(m_max_depth,nullptr);
    _size = 0;
  }

  Node* next_node_depth(std::vector<Node*>& _path, int64_t& _size);

  int m_order;

  int m_max_depth;

  T m_tmp;

  Node* mp_root;

  Log::Logger m_logger;

};

template <typename T>
BTree<T>::BTree(int _order)
  : m_order{_order}
  , m_max_depth{1}
  , m_tmp{}
  , mp_root{nullptr}
  , m_logger{Log::create("BTree")}
{
  mp_root = allocate_node();
}

template <typename T>
BTree<T>::~BTree()
{
  std::list<Node*> node_list;
  std::vector<Node*> path;
  int64_t depth;

  node_depth_init(path, depth);

  while (true)
  {
    Node* p_node = next_node_depth(path, depth);
    if (!p_node)
    {
      break;
    }
    node_list.push_back(p_node);
  }

  for (auto& p_node : node_list)
  {
    deallocate_node(p_node);
  }
}

template <typename T>
void BTree<T>::insert(int64_t _index, T _value)
{
  Node* p_prev = nullptr;
  Node* p_node = mp_root;

  int idepth = 0;

  std::vector<Node*> path(m_max_depth,nullptr);

  while (true)
  {
    Log::debug(m_logger, "Current node: {}", (void*)p_node);

    auto p_indices = p_node->indices();
    auto p_values = p_node->values();
    auto p_children = p_node->children();

    // empty root node, just put it in
    if (p_node->size == 0)
    {
      Log::debug(m_logger, "Inserting in root");

      p_indices[0] = _index;
      p_values[0] = _value;
      p_node->size = 1;
      break;
    }

    // find lower bound
    auto p_lb = std::lower_bound(p_indices, p_indices+p_node->size, _index,
      [](auto a, auto b) { return a <= b; });

    const bool is_last = (p_lb == p_indices+p_node->size);
    const int lb_idx = p_lb-p_indices;

    // if equal, replace
    if (!is_last && *p_lb == _index)
    {
      Log::debug(m_logger, "Replacing value for index {}", _index);
      p_values[lb_idx] = _value;
      break;
    }

    // if found and at max depth, insert
    if (idepth == m_max_depth-1)
    {
      Log::debug(m_logger, "Inserting value into node for index {} at depth {}", _index, idepth);
      insert_node(p_node, path, idepth, _index, _value, nullptr);
      break;
    }

    // If we are not at the bottom of the tree, go to next child
    if (p_children[lb_idx])
    {
      Log::debug(m_logger, "Going to next child");
      p_prev = p_node;
      p_node = p_children[lb_idx];
      path[idepth] = p_prev;
      ++idepth;
      continue;
    }

    // if we are at this point, something went wrong
    throw std::runtime_error("Failed to insert node into tree");

  }
}

template <typename T>
void BTree<T>::remove(int64_t _index)
{
  Node* p_prev = nullptr;
  Node* p_node = mp_root;

  int idepth = 0;

  std::vector<Node*> path(m_max_depth,nullptr);

  while (true)
  {
    Log::debug(m_logger, "Current node: {}", (void*)p_node);

    auto p_indices = p_node->indices();
    auto p_values = p_node->values();
    auto p_children = p_node->children();

    // empty root node, just put it in
    if (p_node->size == 0)
    {
      Log::debug(m_logger, "Empty root");
      break;
    }

    // find lower bound
    auto p_lb = std::lower_bound(p_indices, p_indices+p_node->size, _index,
      [](auto a, auto b) { return a <= b; });

    const bool is_last = (p_lb == p_indices+p_node->size);
    const int lb_idx = p_lb-p_indices;

    // if equal, delete
    if ((lb_idx != 0) && *(p_lb-1) == _index)
    {
      Log::debug(m_logger, "Removing value at index {}", _index);
      remove_node(p_node, path, idepth, _index);
      break;
    }

    // if not equal and at lowest depth, element was not found
    if (idepth == m_max_depth-1)
    {
      Log::debug(m_logger, "Value not found, nothing to remove");
      break;
    }

    // If we are not at the bottom of the tree, go to next child
    if (p_children[lb_idx])
    {
      Log::debug(m_logger, "Going to next child");
      p_prev = p_node;
      p_node = p_children[lb_idx];
      path[idepth] = p_prev;
      ++idepth;
      continue;
    }

    // if we are at this point, something went wrong
    throw std::runtime_error("Failed to remove node into tree");
  }

}

template <typename T>
void BTree<T>::insert_node(Node* _p_node, std::vector<Node*>& _path, int _idepth, int64_t _index, T _value, Node* _p_neighbor)
{
  Log::debug(m_logger, "Insert node {}", (void*)_p_node);

  int64_t* p_indices = _p_node->indices();
  T* p_values = _p_node->values();
  Node** p_children = _p_node->children();

  // find lower bound
  auto p_lb = std::lower_bound(p_indices, p_indices+_p_node->size, _index,
    [](auto a, auto b) { return a <= b; });

  const bool is_last = (p_lb == p_indices+_p_node->size);
  const int lb_idx = p_lb-p_indices;

  // if enough space for insertion and we are the bottom of tree, insert the value
  if (_p_node->size < m_order-1)
  {
    Log::debug(m_logger, "Inserting at pos {} index {} of value {} at depth {} with size {}", 
      lb_idx, _index, _value, _idepth, _p_node->size);

    Log::debug(m_logger, "Indices: {}", fmt::join(p_indices, p_indices+_p_node->size, ","));

    // shift to right
    std::copy_backward(p_indices + lb_idx, p_indices + _p_node->size, 
                       p_indices + _p_node->size + 1);
    std::copy_backward(p_values + lb_idx, p_values + _p_node->size, 
                       p_values + _p_node->size + 1);
    std::copy_backward(p_children + lb_idx, p_children + _p_node->size+1, 
                       p_children + _p_node->size + 2);

    p_indices[lb_idx] = _index;
    p_values[lb_idx] = _value;
    p_children[lb_idx+1] = _p_neighbor;

    _p_node->size++;

    Log::debug(m_logger, "Indices: {}", fmt::join(p_indices, p_indices+_p_node->size, ","));

    return;
  }
  
  // if not enough space for insertion, split
  if (_p_node->size == m_order-1)
  {
    Log::debug(m_logger, "Splitting node");

    // tmp arrays
    std::vector<int64_t> findices(p_indices, p_indices+_p_node->size);
    std::vector<T> fvalues(p_values, p_values+_p_node->size);
    std::vector<Node*> fchildren(p_children, p_children+_p_node->size+1);

    findices.insert(findices.begin()+lb_idx, _index);
    fvalues.insert(fvalues.begin()+lb_idx, _value);
    fchildren.insert(fchildren.begin()+lb_idx+1, _p_neighbor);

    const int split_idx = _p_node->size/2;

    DEBUG_VAR(m_logger, split_idx);
    
    auto p_node2 = allocate_node();

    auto p_indices2 = p_node2->indices();
    auto p_values2 = p_node2->values();
    auto p_children2 = p_node2->children();

    std::copy(findices.begin(), findices.begin()+split_idx, p_indices);
    std::copy(fvalues.begin(), fvalues.begin()+split_idx, p_values);
    std::copy(fchildren.begin(), fchildren.begin()+split_idx, p_children);

    _p_node->size = split_idx;

    Log::debug(m_logger, "Indices Left: {}", 
      fmt::join(findices.begin(), findices.begin()+split_idx, ","));

    std::copy(findices.begin()+split_idx+1, findices.end(), p_indices2);
    std::copy(fvalues.begin()+split_idx+1, fvalues.end(), p_values2);
    std::copy(fchildren.begin()+split_idx+1, fchildren.end(), p_children2);

    p_node2->size = findices.size()-split_idx-1;

    Log::debug(m_logger, "Indices Right: {}", 
      fmt::join(findices.begin()+split_idx+1, findices.end(), ","));

    Log::debug(m_logger, "Index median: {}", findices[split_idx]);

    if (_idepth == 0)
    {
      // if we are at root, create a new root node
      Log::debug(m_logger, "Creating new root");

      Node* p_root_new = allocate_node();
      p_root_new->size = 1;
      p_root_new->indices()[0] = findices[split_idx];
      p_root_new->values()[0] = fvalues[split_idx];
      p_root_new->children()[0] = _p_node;
      p_root_new->children()[1] = p_node2;

      mp_root = p_root_new;

      ++m_max_depth;

      return;
    }
    else 
    {
      insert_node(_path[_idepth-1], _path, _idepth-1, findices[split_idx], fvalues[split_idx], p_node2);
    }
  }
}

template <typename T>
void BTree<T>::remove_node(Node* _p_node, std::vector<Node*>& _path, int _idepth, int64_t _index)
{
  Log::debug(m_logger, "Remove index on node {}", (void*)_p_node);

  int64_t* p_indices = _p_node->indices();
  T* p_values = _p_node->values();
  Node** p_children = _p_node->children();

  // find index
  auto p_lb = std::find(p_indices, p_indices+_p_node->size, _index);

  const bool is_last = (p_lb == p_indices+_p_node->size);
  const int lb_idx = p_lb-p_indices;

  // case 1: node is leaf, node is not empty after delete
  if ((_idepth == m_max_depth-1) && (_p_node->size > 1))
  {
    Log::debug(m_logger, "Deleting at pos {} index {} at depth {} with size {}", 
      lb_idx, _index, _idepth, _p_node->size);

    // shift to left
    std::copy(p_indices + lb_idx + 1, p_indices + _p_node->size, p_indices + lb_idx);
    std::copy(p_values + lb_idx + 1, p_values + _p_node->size, p_values + lb_idx);
    std::copy(p_children + lb_idx + 1, p_children + _p_node->size+1, p_children + lb_idx);

    _p_node->size--;

    Log::debug(m_logger, "Indices: {}", fmt::join(p_indices, p_indices+_p_node->size, ","));

    return;
  }

  // case 2: node is leaf, and will be empty after delete
  if ((_idepth == m_max_depth-1) && (_p_node->size == 1)) 
  {
    Node* p_left_neighbour = get_neighbour(_p_node, _path, _idepth, true);
    Node* p_right_neighbour = get_neighbour(_p_node, _path, _idepth, false);

    Log::debug(m_logger, "Left: {}/Right: {}", (void*)p_left_neighbour, (void*)p_right_neighbour);

    const bool left_valid = p_left_neighbour && p_left_neighbour->size > 1;
    const bool right_valid = p_right_neighbour && p_right_neighbour->size > 1;

    // case 2.1 : neighbours of node have enough indices to borrow
    if ((left_valid || right_valid))
    {
      Node* p_parent = _path[_idepth-1];
      
      if (left_valid)
      {
        Log::debug(m_logger, "Left leaf rotation");
        // put last index from left neighbour into spot nchild-1 in parent, 
        // and put index from spot nchild-1 in current node
        const int nchild = std::find(p_parent->children(), p_parent->children()+p_parent->size+1,
                                    _p_node) - p_parent->children();

        const int64_t left_index = p_left_neighbour->indices()[p_left_neighbour->size-1];
        const T left_value = p_left_neighbour->values()[p_left_neighbour->size-1];
        p_left_neighbour->size--;

        int64_t& parent_index = p_parent->indices()[nchild-1];
        T& parent_value = p_parent->values()[nchild-1];

        p_indices[0] = parent_index;
        p_values[0] = parent_value;

        parent_index = left_index;
        parent_value = left_value;

        return;
      }
      else 
      {
        Log::debug(m_logger, "Right leaf rotation");
        // put first index from right neighbour into spot nchild in parent, 
        // and put index from spot nchild in current node
        const int nchild = std::find(p_parent->children(), p_parent->children()+p_parent->size+1,
                                    _p_node) - p_parent->children();

        const int64_t right_index = p_right_neighbour->indices()[0];
        const T right_value = p_right_neighbour->values()[0];

        // shift left
        std::copy(p_right_neighbour->indices()+1, p_right_neighbour->indices_end(), 
                  p_right_neighbour->indices());

        std::copy(p_right_neighbour->values()+1, p_right_neighbour->values_end(), 
                  p_right_neighbour->values());

        p_right_neighbour->size--;

        int64_t& parent_index = p_parent->indices()[nchild];
        T& parent_value = p_parent->values()[nchild];

        p_indices[0] = parent_index;
        p_values[0] = parent_value;

        parent_index = right_index;
        parent_value = right_value;

        return;
      }
    }
    
    Node* p_parent = _path[_idepth-1];

    // case 2.2 : No suitable neighbours, and more than 1 index in parent
    if (p_parent->size > 1)
    {
      // if this node is leftmost, put first parent index in second child, change parent accordingly
      if (_p_node == p_parent->children()[0])
      {
        Log::debug(m_logger, "Left single-index leaf deletion");

        int64_t parent_idx = p_parent->indices()[0];
        T parent_value = p_parent->values()[0];

        // shift to left
        std::copy(p_parent->indices()+1, p_parent->indices_end(), p_parent->indices());
        std::copy(p_parent->values()+1, p_parent->values_end(), p_parent->values());
        std::copy(p_parent->children()+1, p_parent->children_end(), p_parent->children());

        p_parent->size--;

        // insert value into node
        Node* p_node2 = p_parent->children()[0];

        // shift to right 
        std::copy_backward(p_node2->indices(), p_node2->indices_end(), p_node2->indices_end()+1);
        std::copy_backward(p_node2->values(), p_node2->values_end(), p_node2->values_end()+1);

        p_node2->indices()[0] = parent_idx;
        p_node2->values()[0] = parent_value;
        p_node2->size++;

        deallocate_node(_p_node);

        return;
      }
      // if this is child N of parent, put parent index/value N-1 into end of child N-1
      // then shift parent N to end to N-1 end-1
      else
      {
        Log::debug(m_logger, "Right single-index leaf deletion");

        // get parent position
        int n = std::find(p_parent->children(), p_parent->children_end(), _p_node) 
                - p_parent->children();

        int64_t parent_idx = p_parent->indices()[n-1];
        T parent_value = p_parent->values()[n-1];

        // shift to left
        std::copy(p_parent->indices()+n, p_parent->indices_end(), p_parent->indices()+n-1);
        std::copy(p_parent->values()+n, p_parent->values_end(), p_parent->values()+n-1);
        std::copy(p_parent->children()+n+1, p_parent->children_end(), p_parent->children()+n);

        p_parent->size--;

        // insert value into node
        Node* p_node2 = p_parent->children()[n-1];

        p_node2->indices_end()[0] = parent_idx;
        p_node2->values_end()[0] = parent_value;
        p_node2->size++;

        deallocate_node(_p_node);

        return;
      }
    }
  } // endif case 2

  // case 3: Node is internal, and has more than 1 indices
  if (_idepth != m_max_depth-1 && _p_node->size > 1)
  {
    Log::debug(m_logger, "Removing index in internal node with size > 1");

    // merge right child into left child
    Node* p_left = _p_node->children()[lb_idx];
    Node* p_right = _p_node->children()[lb_idx+1];

    // check for overflow

    std::copy(p_right->values(), p_right->values_end(), p_left->values_end());
    std::copy(p_right->indices(), p_right->indices_end(), p_left->indices_end());
    std::copy(p_right->children(), p_right->children_end(), p_left->children_end());

    p_left->size += p_right->size;

    deallocate_node(p_right);

    // shift parent indices to left
    std::copy(_p_node->indices()+lb_idx+1, _p_node->indices_end(), _p_node->indices()+lb_idx);
    std::copy(_p_node->values()+lb_idx+1, _p_node->values_end(), _p_node->values()+lb_idx);
    std::copy(_p_node->children()+lb_idx+2, _p_node->children_end(), _p_node->children()+lb_idx+1);

    _p_node->size--;

    return;
  }

  throw std::runtime_error("Could not remove index in node");

}

template <typename T>
typename BTree<T>::Node* BTree<T>::get_neighbour(Node* _node, 
                                                 std::vector<Node*>& _path, 
                                                 int _depth,
                                                 bool _left)
{
  if (_node == mp_root)
  {
    return nullptr;
  }

  Node* prev_node = _path[_depth-1];

  Node** p_prev_children = prev_node->children();

  auto it = std::find(p_prev_children, p_prev_children+prev_node->size+1, _node);

  if (it == p_prev_children+prev_node->size+1)
  {
    throw std::runtime_error("Could not get left node, bad child");
  }

  int idx = it - p_prev_children;

  const bool invalid = (_left) ? (idx <= 0) : (idx >= prev_node->size);

  if (invalid)
  {
    return nullptr;
  }
  else 
  {
    return p_prev_children[idx + ((_left) ? -1 : 1)];
  }

}

template <typename T>
typename BTree<T>::Node* BTree<T>::allocate_node()
{
  const int data_size = Node::data_size(m_order);

  DEBUG_VAR(m_logger, data_size);

  auto p_node = reinterpret_cast<Node*>(new uint8_t[data_size]);
  return new (p_node) Node(m_order);
}

template <typename T>
void BTree<T>::deallocate_node(Node* _pnode)
{
  delete  [] reinterpret_cast<uint8_t*>(_pnode);
}

template <typename T>
typename BTree<T>::Node* BTree<T>::next_node_depth(std::vector<Node*>& _path, int64_t& _size)
{
  Log::debug(m_logger, "Next node, size {} maxdepth {}", _size, m_max_depth);

  if (_size < 0)
  {
    return nullptr;
  }

  if (_size == 0 && _path[0] == nullptr)
  {
    _path[0] = mp_root;
    _size = 1;
    return mp_root;
  }  

  if (_size == 0 && _path[0] == mp_root && m_max_depth == 1)
  {
    return nullptr;
  }

  if (_size == m_max_depth)
  {
    return next_node_depth(_path, --_size);
  }

  Node* p_node = _path[_size-1];

  Node* p_next = _path[_size];

  Node** p_children = p_node->children();

  // is the next node in the path inside list?
  Node** it = std::find(p_children, p_children + p_node->size+1, p_next);

  if (it == p_children+p_node->size+1)
  {
    Log::debug(m_logger, "Did not find next child. First time visiting.");
    // first time visiting, go to first child
    _path[_size] = p_children[0];
    _size++;

    return _path[_size-1];
  }
  else if (it == p_children+p_node->size)
  {
    Log::debug(m_logger, "Next is last child, going up");

    // we are at the root, so finish
    if (_size == 1)
    {
      --_size;
      return nullptr;
    }

    // last child was visited, go back up
    return next_node_depth(_path, --_size);
  }
  else 
  {
    // found it, so go to next child
    Log::debug(m_logger, "Found child at {} going to next", (it - p_children));
    _path[_size] = p_children[it - p_children + 1];
    _size++;

    return _path[_size-1];
  }

}

template <typename T>
std::string BTree<T>::make_pretty()
{
  std::string out;

  std::vector<Node*> path;
  int64_t depth;

  node_depth_init(path, depth);

  while (true)
  {
    Node* p_node = next_node_depth(path, depth);
    if (!p_node)
    {
      break;
    }

    auto indices = p_node->indices();

    Log::debug(m_logger, "Got indices {}", fmt::join(indices, indices+p_node->size,","));

    std::string delimiter = std::string(",\n") + std::string(depth, ' ') + " ";

    out += fmt::format("{} {}\n", std::string(depth, '-'),
                       fmt::join(indices, indices+p_node->size,delimiter));
  }
  
  return out;

}

} // namespace Shtensor

#endif