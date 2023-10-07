#ifndef SHTENSOR_BTREE_H
#define SHTENSOR_BTREE_H

#include "Definitions.h"
#include "Logger.h"
#include "Utils.h"

#include <functional>
#include <list>

// TO DO: 
// - Remove unnecessary finds (use path)
// - Init function for large sorted array
// - Make it less garbage
// - Look at how to use MemoryPool


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

    uint8_t* data;

    static int64_t data_size(int _order)
    {
      const int ssize = 
          _order*SSIZEOF(Node*) // pointers to children
        + (_order-1)*SSIZEOF(int64_t) // indices
        + (_order-1)*SSIZEOF(T) // values
      ;

      // round to next multiple of 16 for correct alignment
      return Utils::round_next_multiple(ssize, 16);
    } 

    Node** children()
    {
      return reinterpret_cast<Node**>(data);
    }

    Node** children_end()
    {
      return children() + size + 1;
    }

    int64_t* indices()
    {
      return reinterpret_cast<int64_t*>(data + order*SSIZEOF(Node*));
    }

    int64_t* indices_end()
    {
      return indices() + size;
    }

    T* values()
    {
      return reinterpret_cast<T*>(data + order*SSIZEOF(Node*) + (order-1)*SSIZEOF(int64_t));
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

  BTree(int _order, 
        int64_t* _pindices_begin, 
        int64_t* _pindices_end, 
        T* _pvalues_begin,
        T* _pvalues_end);

  ~BTree();

  void insert(int64_t _index, T _value);

  void erase(int64_t _index);

  void fill_ordered_array(int64_t* _pindices, T* _pvalues);

  bool is_valid(); 

  int64_t size();

  std::string make_pretty();

  class Iterator
  {
   public:

    using iterator_category = std::forward_iterator_tag;
    using difference_type = int64_t;
    using value_type = T;
    using pointer = T*;  
    using reference = T&;

    Iterator(const BTree& _tree)
      : m_tree{_tree}
      , m_nodes{}
      , m_path{}
      , m_depth{}
      , mp_index{nullptr}
      , mp_value{nullptr}
    {
      m_tree.inorder_begin(m_nodes, m_path, m_depth, &mp_index, &mp_value);
    }

    Iterator(const Iterator& _iter) = default;

    Iterator(Iterator&& _iter) = default;

    Iterator& operator=(const Iterator& _iter) = default;

    Iterator& operator=(Iterator&& _iter) = default;

    ~Iterator() {}

    Iterator get_end()
    {
      Iterator out;

    }

    int64_t key() 
    {
      return *mp_index;
    }

    reference operator*()
    {
      return *mp_value;
    }

    pointer operator->()
    {
      return mp_value;
    }

    Iterator& operator++() 
    { 
      m_tree.inorder_next(m_nodes, m_path, m_depth, &mp_index, &mp_value);
      return *this;
    }
    
    Iterator operator++(int)
    {
      Iterator tmp_iter(*this);
      m_tree.inorder_next(m_nodes, m_path, m_depth, &mp_index, &mp_value);
      return tmp_iter;
    }
    
    // BlockIteratorDetail& operator+=(difference_type _dt)
    // {
    //   m_idx += _dt;
    //   update_block();
    //   return *this;
    // }
    
    // BlockIteratorDetail operator+(difference_type _dt) const
    // {
    //   BlockIteratorDetail tmp_iter(*this);
    //   tmp_iter += _dt;
    //   tmp_iter.update_block();
    //   return tmp_iter;
    // }

    // BlockIteratorDetail& operator--() 
    // { 
    //   m_idx--; 
    //   update_block();
    //   return *this;
    // }
    
    // BlockIteratorDetail operator--(int)
    // {
    //   BlockIteratorDetail tmp_iter(*this);
    //   m_idx--;
    //   update_block();
    //   return tmp_iter;
    // }
    
    // BlockIteratorDetail& operator-=(difference_type _dt)
    // {
    //   m_idx -= _dt;
    //   update_block();
    //   return *this;
    // }
    
    // BlockIteratorDetail operator-(difference_type _dt) const
    // {
    //   BlockIteratorDetail tmp_iter(*this);
    //   tmp_iter -= _dt;
    //   tmp_iter.update_block();
    //   return tmp_iter;
    // }
    
    // difference_type operator-(const BlockIteratorDetail& _iter) const
    // {
    //   return m_idx - _iter.m_idx;
    // }

    // bool operator<(const BlockIteratorDetail& _iter) const 
    // {
    //   return m_idx < _iter.m_idx;
    // }

    // bool operator>(const BlockIteratorDetail& _iter) const 
    // {
    //   return m_idx > _iter.m_idx;
    // }

    // bool operator<=(const BlockIteratorDetail& _iter) const 
    // {
    //   return m_idx <= _iter.m_idx;
    // }

    // bool operator>=(const BlockIteratorDetail& _iter) const 
    // {
    //   return m_idx >= _iter.m_idx;
    // }

    bool operator==(const Iterator& _iter) const 
    {
      return (mp_index == _iter.mp_index) && (m_depth == _iter.m_depth);
    }

    bool operator!=(const Iterator& _iter) const 
    {
      return mp_index != _iter.mp_index;
    }

    friend class BTree;

   protected:

    Iterator(const BTree& _tree, 
             const std::vector<Node*>& _nodes,  
             const std::vector<int>& _path,
             int _idepth,
             int64_t* _pindex,
             T* _pvalue)
      : m_tree{_tree}
      , m_nodes{_nodes}
      , m_path{_path}
      , m_depth{_idepth}
      , mp_index{_pindex}
      , mp_value{_pvalue}
    {
    }

    const BTree& m_tree;

    std::vector<Node*> m_nodes;
    std::vector<int> m_path;
    int m_depth;
    int64_t* mp_index;
    T* mp_value;

  };

  Iterator begin() { return Iterator(*this); }

  Iterator end() { return Iterator(*this, {}, {}, 0, mp_root->indices_end(), nullptr); }

  T* find(int64_t _index);
 
 protected:

  void node_depth_init(std::vector<Node*>& _path, int64_t& _size)
  {
    _path = std::vector<Node*>(m_max_depth,nullptr);
    _size = 0;
  }

  Node* next_node_depth(std::vector<Node*>& _path, 
                        int64_t& _size);

  void inorder_begin(std::vector<Node*>& _predecessors, 
                     std::vector<int>& _path, 
                     int& _idepth,
                     int64_t** _ppindex,
                     T** _ppvalue) const;

  void inorder_next(std::vector<Node*>& _predecessors, 
                    std::vector<int>& _path, 
                    int& _idepth,
                    int64_t** _ppindex,
                    T** _ppvalue) const;

  void fill_ordered_array_node(Node* _pnode, 
                               int64_t* _pindices, 
                               T* _pvalues, 
                               int64_t& _pos);

  void insert_node(Node* _node, 
                   std::vector<Node*>& _path, 
                   std::vector<int>& _positions,
                   int _depth, 
                   int64_t _index, 
                   T _value, 
                   Node* _p_left, 
                   Node* _p_right);

  void insert_into_node(Node* _node, 
                        const std::vector<int64_t>& _indices, 
                        const std::vector<T>& _values, 
                        const std::vector<Node*>& _p_children,
                        int _pos);

  void split_node(Node* _node, 
                  std::vector<Node*>& _path, 
                  std::vector<int>& _pos, 
                  int _depth);

  void erase_node(Node* _node, 
                  std::vector<Node*>& _path, 
                  int _depth, 
                  int64_t _index);

  void rebalance_node(Node* _p_node, 
                      std::vector<Node*>& _path, 
                      int _idepth);

  void merge_node(Node* _node, 
                  std::vector<Node*>& _path, 
                  int _idepth);

  Node* get_neighbour(Node* _node, 
                      std::vector<Node*>& _path, 
                      int _depth, 
                      bool _left);

  Node* allocate_node();

  void deallocate_node(Node* _pnode);

  std::string make_pretty_node(Node* _pnode, 
                               const std::string& _prefix, 
                               bool _last);

  int m_order;

  int m_max_depth;

  Node* mp_root;

  Log::Logger m_logger;

};

template <typename T>
BTree<T>::BTree(int _order)
  : m_order{_order}
  , m_max_depth{0}
  , mp_root{nullptr}
  , m_logger{Log::create("BTree")}
{
  mp_root = allocate_node();
}

template <typename T>
BTree<T>::BTree(int _order, 
                int64_t* _pindices_begin, 
                int64_t* _pindices_end, 
                T* _pvalues_begin,
                T* _pvalues_end)
  : m_order{_order}
  , m_max_depth{0}
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
  std::vector<int> pos(m_max_depth,-1);

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
      m_max_depth = 1;
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
      insert_node(p_node, path, pos, idepth, _index, _value, nullptr, nullptr);
      break;
    }

    // If we are not at the bottom of the tree, go to next child
    if (p_children[lb_idx])
    {
      Log::debug(m_logger, "Going to next child");
      p_prev = p_node;
      p_node = p_children[lb_idx];
      path[idepth] = p_prev;
      pos[idepth+1] = lb_idx;
      ++idepth;
      continue;
    }

    // if we are at this point, something went wrong
    throw std::runtime_error("Failed to insert node into tree");

  }
}

template <typename T>
void BTree<T>::erase(int64_t _index)
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
      erase_node(p_node, path, idepth, _index);
      break;
    }

    // if not equal and at lowest depth, element was not found
    if (idepth == m_max_depth-1)
    {
      Log::debug(m_logger, "Value not found, nothing to erase");
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
    throw std::runtime_error("Failed to erase node into tree");
  }

}

template <typename T>
void BTree<T>::insert_node(Node* _p_node, 
                           std::vector<Node*>& _path, 
                           std::vector<int>& _pos,
                           int _idepth, 
                           int64_t _index, 
                           T _value, 
                           Node* _p_left,
                           Node* _p_right)
{
  Log::debug(m_logger, "Insert node {}", (void*)_p_node);

  int64_t* p_indices = _p_node->indices();
  T* p_values = _p_node->values();
  Node** p_children = _p_node->children();

  // find lower bound
  auto p_lb = std::lower_bound(p_indices, p_indices+_p_node->size, _index,
    [](auto a, auto b) { return a <= b; });

  const int lb_idx = p_lb-p_indices;

  Log::debug(m_logger, "Inserting at pos {} index {} of value {} at depth {} with size {}", 
             lb_idx, _index, _value, _idepth, _p_node->size);

  Log::debug(m_logger, "Indices: {}", fmt::join(_p_node->indices(), _p_node->indices_end(), ","));

  insert_into_node(_p_node, {_index}, {_value}, {_p_left, _p_right}, lb_idx);

  Log::debug(m_logger, "Indices: {}", fmt::join(_p_node->indices(), _p_node->indices_end(), ","));
  
  if (_p_node->size == m_order)
  {
    // overflow occurred
    Log::debug(m_logger, "Overflow");

    split_node(_p_node, _path, _pos, _idepth);
  } 

  return;

}

template <typename T>
void BTree<T>::insert_into_node(Node* _pnode, 
                                const std::vector<int64_t>& _indices, 
                                const std::vector<T>& _values, 
                                const std::vector<Node*>& _children,
                                int _pos)
{
  const int new_size = _pnode->size + Utils::ssize(_indices);

  if (_pnode->size == m_order-1)
  {
    uint8_t* p_data = new uint8_t[Node::data_size(new_size+1)];

    Node** p_children = reinterpret_cast<Node**>(p_data);
    int64_t* p_indices = reinterpret_cast<int64_t*>(p_children+new_size+1);
    T* p_values = reinterpret_cast<T*>(p_indices+new_size);

    std::copy(_pnode->indices(), _pnode->indices_end(), p_indices);
    std::copy(_pnode->values(), _pnode->values_end(), p_values);
    std::copy(_pnode->children(), _pnode->children_end(), p_children);

    delete [] _pnode->data;
    _pnode->order = new_size+1;
    _pnode->data = p_data;
  }

  // shift to right
  std::copy_backward(_pnode->indices()+_pos, _pnode->indices_end(), _pnode->indices()+new_size);
  std::copy_backward(_pnode->values()+_pos, _pnode->values_end(), _pnode->values()+new_size);
  std::copy_backward(_pnode->children()+_pos+1, _pnode->children_end(), 
                     _pnode->children()+new_size+1);
  
  std::copy(_indices.begin(), _indices.end(), _pnode->indices()+_pos);
  std::copy(_values.begin(), _values.end(), _pnode->values()+_pos);
  std::copy(_children.begin(), _children.end(), _pnode->children()+_pos);

  _pnode->size = new_size;
  
}

template <typename T>
void BTree<T>::split_node(Node* _p_node, 
                          std::vector<Node*>& _path,
                          std::vector<int>& _pos,
                          int _idepth)
{
  Log::debug(m_logger, "Splitting node");

  const int split_idx = _p_node->size/2;

  DEBUG_VAR(m_logger, split_idx);
  
  Node* p_node_left = allocate_node();
  Node* p_node_right = allocate_node();

  std::copy(_p_node->indices(), _p_node->indices()+split_idx, p_node_left->indices());
  std::copy(_p_node->values(), _p_node->values()+split_idx, p_node_left->values());
  std::copy(_p_node->children(), _p_node->children()+split_idx+1, p_node_left->children());

  p_node_left->size = split_idx;

  Log::debug(m_logger, "Indices Left: {}", 
    fmt::join(p_node_left->indices(), p_node_left->indices_end(), ","));

  std::copy(_p_node->indices()+split_idx+1, _p_node->indices_end(), p_node_right->indices());
  std::copy(_p_node->values()+split_idx+1, _p_node->values_end(), p_node_right->values());
  std::copy(_p_node->children()+split_idx+1, _p_node->children_end(), p_node_right->children());

  p_node_right->size = _p_node->size-split_idx-1;

  Log::debug(m_logger, "Indices Right: {}", 
    fmt::join(p_node_right->indices(), p_node_right->indices_end(), ","));

  Log::debug(m_logger, "Index median: {}", _p_node->indices()[split_idx]);

  if (_idepth == 0)
  {
    // if we are at root, create a new root node
    Log::debug(m_logger, "Creating new root");

    Node* p_root_new = allocate_node();
    p_root_new->size = 1;
    p_root_new->indices()[0] = _p_node->indices()[split_idx];
    p_root_new->values()[0] = _p_node->values()[split_idx];
    p_root_new->children()[0] = p_node_left;
    p_root_new->children()[1] = p_node_right;

    mp_root = p_root_new;

    ++m_max_depth;

  }
  else 
  {
    insert_node(_path[_idepth-1], _path, _pos, _idepth-1, _p_node->indices()[split_idx], 
                _p_node->values()[split_idx], p_node_left, p_node_right);
  }

  deallocate_node(_p_node);

}

template <typename T>
void BTree<T>::erase_node(Node* _p_node, std::vector<Node*>& _path, int _idepth, int64_t _index)
{
  Log::debug(m_logger, "erase index {} on node {}", _index, (void*)_p_node);

  int64_t* p_indices = _p_node->indices();
  T* p_values = _p_node->values();
  Node** p_children = _p_node->children();

  // find index
  auto p_lb = std::find(p_indices, p_indices+_p_node->size, _index);
  const int lb_idx = p_lb-p_indices;

  // case 1: node is leaf
  if ((_idepth == m_max_depth-1))
  {
    Log::debug(m_logger, "Deleting at pos {} index {} at depth {} with size {}", 
               lb_idx, _index, _idepth, _p_node->size);

    // shift to left
    std::copy(_p_node->indices()+lb_idx+1, _p_node->indices_end(), _p_node->indices()+lb_idx);
    std::copy(_p_node->values()+lb_idx+1, _p_node->values_end(), _p_node->values()+lb_idx);

    _p_node->size--;

    Log::debug(m_logger, "Indices: {}", fmt::join(p_indices, p_indices+_p_node->size, ","));

    // done if node is not empty
    if (_p_node->size > 0) return;

    _path[_idepth] = _p_node;
    rebalance_node(_p_node, _path, _idepth);

    return;
    
  } // endif leaf node

  // case 2: Node is internal
  if (_idepth != m_max_depth-1)
  {
    Log::debug(m_logger, "Removing index in internal node");

    // Take largest index from left child and put it in deleted position
    Node* p_offspring = _p_node->children()[lb_idx];
    _path[_idepth] = _p_node;
    int jdepth = _idepth+1;
    
    while (jdepth != m_max_depth-1)
    {
      _path[jdepth] = p_offspring;
      p_offspring = p_offspring->children_end()[-1];
      ++jdepth;
    }

    _path[jdepth] = p_offspring;

    Log::debug(m_logger, "Largest index on left of node: {}", p_offspring->indices_end()[-1]);

    _p_node->indices()[lb_idx] = p_offspring->indices_end()[-1];
    _p_node->values()[lb_idx] = p_offspring->values_end()[-1];

    p_offspring->size--;

    if (p_offspring->size == 0)
    {
      merge_node(p_offspring, _path, jdepth);
    }

    return;
  }

  throw std::runtime_error("Could not erase index in node");

}

template <typename T>
void BTree<T>::rebalance_node(Node* _p_node, std::vector<Node*>& _path, int _idepth)
{
  Log::debug(m_logger, "Rebalancing tree from node {} at depth {}", (void*)_p_node, _idepth);

  Node* p_parent = _path[_idepth-1];
  Node* p_left_neighbour = get_neighbour(_p_node, _path, _idepth, true);
  Node* p_right_neighbour = get_neighbour(_p_node, _path, _idepth, false);

  const bool left_valid = p_left_neighbour && p_left_neighbour->size > 1;
  const bool right_valid = p_right_neighbour && p_right_neighbour->size > 1;

  // case 1 : left neighbour is larger than minimum
  if (left_valid) 
  {
    Log::debug(m_logger, "Left leaf rotation");
    // put last index from left neighbour into spot nchild-1 in parent, 
    // and put index from spot nchild-1 in current node
    // put last child of left neighbour into first child of current node
    const int nchild = std::find(p_parent->children(), p_parent->children()+p_parent->size+1,
                                _p_node) - p_parent->children();

    const int64_t left_index = p_left_neighbour->indices()[p_left_neighbour->size-1];
    const T left_value = p_left_neighbour->values()[p_left_neighbour->size-1];
    Node* p_neighbour_child = p_left_neighbour->children_end()[-1];
    p_left_neighbour->size--;

    int64_t& parent_index = p_parent->indices()[nchild-1];
    T& parent_value = p_parent->values()[nchild-1];

    _p_node->indices()[0] = parent_index;
    _p_node->values()[0] = parent_value;
    _p_node->children()[1] = _p_node->children()[0];
    _p_node->children()[0] = p_neighbour_child;
    _p_node->size++;

    parent_index = left_index;
    parent_value = left_value;

    return;
  }

  // case 1.2 : right neighbour is large enough
  if (right_valid) 
  {
    Log::debug(m_logger, "Right  rotation");
    // put first index from right neighbour into spot nchild in parent, 
    // and put index from spot nchild in current node
    // put first child of rightt neighbour into last child of current node
    const int nchild = std::find(p_parent->children(), p_parent->children_end(),_p_node) 
                        - p_parent->children();

    const int64_t right_index = p_right_neighbour->indices()[0];
    const T right_value = p_right_neighbour->values()[0];
    Node* right_child = p_right_neighbour->children()[0];

    // shift left
    std::copy(p_right_neighbour->indices()+1, p_right_neighbour->indices_end(), 
              p_right_neighbour->indices());
    std::copy(p_right_neighbour->values()+1, p_right_neighbour->values_end(), 
              p_right_neighbour->values());
    std::copy(p_right_neighbour->children()+1, p_right_neighbour->children_end(),
              p_right_neighbour->children());

    p_right_neighbour->size--;

    int64_t& parent_index = p_parent->indices()[nchild];
    T& parent_value = p_parent->values()[nchild];

    _p_node->indices()[0] = parent_index;
    _p_node->values()[0] = parent_value;
    _p_node->children()[1] = right_child;
    _p_node->size++;

    parent_index = right_index;
    parent_value = right_value;

    return;
  }
  
  // case 1.3 : No suitable neighbour, merge
  if (!left_valid && !right_valid)
  {
    _path[_idepth] = _p_node;
    merge_node(_p_node, _path, _idepth);
  }

  return;
}

template <typename T>
void BTree<T>::merge_node(Node* _p_node, std::vector<Node*>& _path, int _idepth)
{
  Log::debug(m_logger, "Merging node {}", (void*)_p_node);

  // _p_node is assumed empty, children()[0] may contain a single node from a previous merge
  // Merge parent with left or right neighbor
  //Node* p_left = get_neighbour(_p_node, _path, _idepth, true);
  //Node* p_right = (p_left) ? nullptr : get_neighbour(_p_node, _path, _idepth, false);
  Node* p_parent = _path[_idepth-1]; 

  // Find index of node in parent
  const int idx = std::find(p_parent->children(), p_parent->children_end(), _p_node)
                  - p_parent->children();

  const int sibling_idx = (idx == 0) ? 1 : idx-1;

  Log::debug(m_logger, "Merging sibling with index {} to empty sibling {}", sibling_idx, idx);

  Node* p_sibling = p_parent->children()[sibling_idx];

  if (_p_node->size != 0)
  {
    throw std::runtime_error("Cannot merge, node does not have size 0");
  }

  // pull down part of parent into sibling
  if (idx == 0)
  {
    Log::debug(m_logger, "Sibling has size {}", p_sibling->size);

     // shift stuff in sibling to right
    std::copy_backward(p_sibling->indices(), p_sibling->indices_end(), p_sibling->indices_end()+1);
    std::copy_backward(p_sibling->values(), p_sibling->values_end(), p_sibling->values_end() + 1);
    std::copy_backward(p_sibling->children(), p_sibling->children_end(), p_sibling->children_end() + 1);

    // copy index from parent to sibling
    p_sibling->indices()[0] = p_parent->indices()[0];
    p_sibling->values()[0] = p_parent->values()[0];
    p_sibling->children()[0] = _p_node->children()[0];

    p_sibling->size++;
    p_parent->size--;

    p_parent->children()[0] = p_sibling;
  }
  else
  {
    // put parent index into sibling
    p_sibling->indices_end()[0] = p_parent->indices()[idx-1];
    p_sibling->values_end()[0] = p_parent->values()[idx-1];
    p_sibling->children_end()[0] = _p_node->children()[0];

    // shift stuff in parent to left
    if (p_parent->size > 1)
    {
      std::copy(p_parent->indices()+idx, p_parent->indices_end(), p_parent->indices()+idx-1);
      std::copy(p_parent->values()+idx, p_parent->values_end(), p_parent->values()+idx-1);
      std::copy(p_parent->children()+idx+1, p_parent->children_end(), p_parent->children()+idx);
    }
  
    p_sibling->size++;
    p_parent->size--;
  }

  deallocate_node(_p_node);

  if (p_parent != mp_root && p_parent->size == 0)
  {
    rebalance_node(p_parent, _path, _idepth-1);
  }
  else if (p_parent == mp_root)
  {
    deallocate_node(p_parent);
    mp_root = p_sibling;
    m_max_depth--;
  }
  
  return;

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

  auto p_node = new Node(m_order);
  p_node->data = new uint8_t[data_size]();

  return p_node;
}

template <typename T>
void BTree<T>::deallocate_node(Node* _pnode)
{
  delete  [] _pnode->data;
  delete _pnode;
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
std::string BTree<T>::make_pretty_node(Node* _pnode, 
                                       const std::string& _prefix, 
                                       bool _last) 
{
    std::string out;

    out += _prefix;
    out += (_last ? " └──" : " ├──");
    out += fmt::format("({})[{}]\n", 
                       fmt::join(_pnode->indices(), _pnode->indices_end(), ","),
                       (void*)_pnode);

    // Get the new prefix for children
    std::string new_prefix = _prefix + (_last ? "   " : " │ ");

    if (_pnode->children()[0] == nullptr)
    {
      return out;
    }

    // Recursively print the children
    for (auto it = _pnode->children(); it != _pnode->children_end(); ++it) 
    {
      out += make_pretty_node(*it, new_prefix, it+1 == _pnode->children_end());
    }

    return out;
}

template <typename T>
std::string BTree<T>::make_pretty()
{
  return make_pretty_node(mp_root, "", false);
}

template <typename T>
bool BTree<T>::is_valid()
{
  std::vector<Node*> path = {};
  int64_t depth;

  node_depth_init(path, depth);

  while (true)
  {
    Node* p_node = next_node_depth(path, depth);

    if (!p_node)
    {
      break;
    }

    if ((p_node != mp_root && p_node->size <= 0) || p_node->size > m_order-1)
    {
      Log::error(m_logger, "Bad size on node {}. Got {}", (void*)p_node, p_node->size);
      return false;
    }

    if (!std::is_sorted(p_node->indices(), p_node->indices_end()))
    {
      Log::error(m_logger, "Indices are not ordered");
      return false;
    }

    if (depth != m_max_depth 
        && std::find(p_node->children(), p_node->children_end(), nullptr) != p_node->children_end())
    {
      Log::error(m_logger, "Invalid child");
      return false;
    }

  }

  return true;
}

template <typename T>
int64_t BTree<T>::size()
{
  int64_t size = 0;

  fill_ordered_array_node(mp_root, nullptr, nullptr, size);

  return size;
}

template <typename T>
void BTree<T>::fill_ordered_array(int64_t* _pindices, T* _pvalues)
{
  int64_t pos = 0;
  fill_ordered_array_node(mp_root, _pindices, _pvalues, pos);
}

template <typename T>
void BTree<T>::fill_ordered_array_node(Node* _pnode, int64_t* _pindices, T* _pvalues, int64_t& _pos)
{
  if (_pnode)
  {
    for (int i = 0; i < _pnode->size+1; ++i)
    {
        fill_ordered_array_node(_pnode->children()[i], _pindices, _pvalues, _pos);

        if (i < _pnode->size)
        {
          if (_pindices)
          {
            _pindices[_pos] = _pnode->indices()[i];
          }

          if (_pvalues)
          {
            _pvalues[_pos] = _pnode->values()[i];
          }
          
          ++_pos;
        }
    }
  }
}

template <typename T>
void BTree<T>::inorder_begin(std::vector<Node*>& _predecessors, 
                             std::vector<int>& _path, 
                             int& _idepth,
                             int64_t** _pindex,
                             T** _pvalue) const
{
  _predecessors.clear();
  _path.clear();
  _idepth = 0;

  _predecessors.resize(m_max_depth, nullptr);
  _path.resize(m_max_depth, -1);

  // go to leftmost element
  Node* p_node = mp_root;
  _path[0] = -1;
  _predecessors[0] = mp_root;

  while (p_node->children()[0])
  {
    p_node = p_node->children()[0];
    ++_idepth;
    _path[_idepth] = 0;
    _predecessors[_idepth] = p_node;
  }

  *_pindex = p_node->indices();
  *_pvalue = p_node->values();

  return;
}

template <typename T>
void BTree<T>::inorder_next(std::vector<Node*>& _predecessors, 
                            std::vector<int>& _path, 
                            int& _idepth,
                            int64_t** _pindex,
                            T** _pvalue) const
{
  // case 1 : Leaf node and not last element
  if (_idepth == m_max_depth-1 && *_pindex+1 != _predecessors[_idepth]->indices_end())
  {
    // go to next element
    *_pindex = *_pindex+1;
    *_pvalue = *_pvalue+1;
    return;
  }

  // case 2 : Leaf node and last element
  // or internal node and last+1 element
  if ((_idepth == m_max_depth-1 && *_pindex+1 == _predecessors[_idepth]->indices_end())
    || (_idepth != m_max_depth-1 && *_pindex == _predecessors[_idepth]->indices_end()))
  {
    // return end if past mp_root
    if (_idepth == 0)
    {
      *_pindex = mp_root->indices_end();
      *_pvalue = nullptr;
      return;
    }

    // return to parent and increase index
    *_pindex = _predecessors[_idepth-1]->indices()+_path[_idepth];
    *_pvalue = _predecessors[_idepth-1]->values()+_path[_idepth];

    --_idepth;

    // if last element in parent....
    if (*_pindex == _predecessors[_idepth]->indices_end())
    {
      inorder_next(_predecessors, _path, _idepth, _pindex, _pvalue);
    }
    
    return;
  }

  // case 3 : Internal node and not last+1 element
  if (_idepth != m_max_depth-1 && *_pindex < _predecessors[_idepth]->indices_end())
  {
    // go to left-most value in next child
    const int next_child = _path[_idepth+1]+1;

    Node* p_node = _predecessors[_idepth]->children()[next_child];
    ++_idepth;
    _path[_idepth] = next_child;
    _predecessors[_idepth] = p_node;

    while (p_node->children()[0])
    {
      ++_idepth;
      p_node = p_node->children()[0];
      _predecessors[_idepth] = p_node;
      _path[_idepth] = 0;
    }

    *_pindex = p_node->indices();
    *_pvalue = p_node->values();

    return;
  }

  throw std::runtime_error("Could not go to next element");

}

template <typename T>
T* BTree<T>::find(int64_t _index)
{
  Node* p_node = mp_root;
  
  while (p_node)
  {
    // find lower bound
    auto p_lb = std::lower_bound(p_node->indices(), p_node->indices_end(), _index,
      [](auto a, auto b) { return a <= b; });

    int pos = p_lb - p_node->indices();

    // element was found
    if (p_lb < p_node->indices_end() && *p_lb == _index)
    {
      return p_node->values() + pos;
    }

    // element was not found, and we are in internal node
    if (p_node->children()[0])
    {
      p_node = p_node->children()[pos];
      continue;
    }

    // element was not found, and we are in leaf
    if (!p_node->children()[0])
    {
      return nullptr;
    }
  }

  return nullptr;

}


} // namespace Shtensor

#endif