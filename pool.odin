// Pool (Public for users who want to re-use it as-is for other resources).
// ----------------------------------------------------------------------------
// The pool is a simple resource management system.
//
// The pool work like this, there is a fixed number of slots (defined at pool
// creation) that can be acquired and released. Each slot has an incrementing
// generation counter, which is used to generate unique ids for each slot.
//
// When a slot is released, its generation counter is incremented, so that any
// ids generated from that slot will be invalid until the slot is acquired
// again.
package sdl_painter

POOL_INVALID_SLOT :: 0
POOL_SLOT_SHIFT   :: 16
POOL_SLOT_MASK    :: ((1 << POOL_SLOT_SHIFT) - 1)

Pool :: struct {
	size:           uint, // total number of slots in the pool (counting the invalid slot)
	counters:       [^]u32, // incrementing generation counters for each slot
	free_stack:     [^]i32, // stack of free slots
	free_stack_top: i32, // index of the top of the free queue
}

// Create a pool with the specified number of slots (not counting the invalid
// slot).
CreatePool :: proc(number_of_slots: int) -> ^Pool {
	pool := new(Pool)

	// +1 since slot 0 is reserved for invalid slot
	pool.size = uint(number_of_slots + 1)
	pool.free_stack_top = 0
	pool.counters = make([^]u32, pool.size)
	pool.free_stack = make([^]i32, number_of_slots)

	for i := int(pool.size) - 1; i > 0; i -= 1 {
		pool.free_stack[pool.free_stack_top] = i32(i)
		pool.free_stack_top += 1
		pool.counters[i] = 0
	}

	return pool
}

// Destroy a pool and free its resources.
DestroyPool :: proc(pool: ^Pool) {
	free(pool.counters)
	free(pool.free_stack)
	free(pool)
}

// Acquire a slot from the pool and return its index. Returns
// POOL_INVALID_SLOT if no more slots are available.
AcquirePoolSlot :: proc(pool: ^Pool) -> i32 {
	assert(pool != nil)
	assert(pool.free_stack != nil)

	if pool.free_stack_top > 0 {
		pool.free_stack_top -= 1
		return pool.free_stack[pool.free_stack_top] // Get a slot from the free queue
	} else {
		return POOL_INVALID_SLOT // No more slots available
	}
}

// Release a slot back to the pool, making it available for future
// acquisitions.
ReleasePoolSlot :: proc(pool: ^Pool, slot: i32) {
	assert(slot > POOL_INVALID_SLOT && int(slot) < int(pool.size))
	assert(pool != nil)
	assert(pool.free_stack != nil)
	assert(pool.free_stack_top < i32(pool.size))

	pool.free_stack[pool.free_stack_top] = slot
	pool.free_stack_top += 1

	assert(pool.free_stack_top <= i32(pool.size))
}

// Generate a unique id for a slot in the pool using its index and generation
// counter.
GeneratePoolId :: proc(pool: ^Pool, slot: i32) -> u32 {
	assert(pool != nil)
	assert(slot > POOL_INVALID_SLOT && int(slot) < int(pool.size))
	assert(pool.counters != nil)

	counter := pool.counters[slot] + 1 // increment generation
	pool.counters[slot] = counter

	id := (counter << POOL_SLOT_SHIFT) | (u32(slot) & POOL_SLOT_MASK)

	return id
}

// Extract the slot index from a generated id.
PoolIdToSlot :: proc(id: u32) -> i32 {
	slot := i32(id & POOL_SLOT_MASK)
	return slot
}
