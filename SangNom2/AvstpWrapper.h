/*****************************************************************************

        AvstpWrapper.cpp
        Author: Laurent de Soras, 2012

--- Legal stuff ---

This program is free software. It comes without any warranty, to
the extent permitted by applicable law. You can redistribute it
and/or modify it under the terms of the Do What The Fuck You Want
To Public License, Version 2, as published by Sam Hocevar. See
http://sam.zoy.org/wtfpl/COPYING for more details.

*Tab=3***********************************************************************/


#ifndef __AVSTP_WRAPPER__
#define __AVSTP_WRAPPER__

#include	"avstp.h"

#include	<memory>



class AvstpWrapper
{

/*\\\ PUBLIC \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

public:

	virtual	~AvstpWrapper();

	static AvstpWrapper &getInstance();

	avstp_TaskDispatcher* createDispatcher();
	void destroyDispatcher(avstp_TaskDispatcher *dispatcher);
	int numberOfThreads();
	int enqueue(avstp_TaskDispatcher *dispatcher, avstp_TaskPtr task, void *userData);

 //   template<class T>
	//int enqueue(avstp_TaskDispatcher *dispatcher, T &&function) {
 //       this->enqueue(dispatcher, function, nullptr);
 //   }
	int	waitCompletion(avstp_TaskDispatcher *dispatcher);



/*\\\ PROTECTED \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

protected:
    AvstpWrapper();



/*\\\ PRIVATE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

private:

	template <class T>
	void resolveName (T &fnc_ptr, const char *name_0);

	void assign_normal ();
	void assign_fallback ();

	static int fallback_get_interface_version_ptr ();
	static avstp_TaskDispatcher * fallback_create_dispatcher_ptr ();
	static void	fallback_destroy_dispatcher_ptr (avstp_TaskDispatcher *td_ptr);
	static int fallback_get_nbr_threads_ptr ();
	static int fallback_enqueue_task_ptr (avstp_TaskDispatcher *td_ptr, avstp_TaskPtr task_ptr, void *user_data_ptr);
	static int fallback_wait_completion_ptr (avstp_TaskDispatcher *td_ptr);

    decltype(&avstp_create_dispatcher) createDispatcherPtr_;
    decltype(&avstp_get_nbr_threads) numberOfThreadsPtr_;
    decltype(&avstp_destroy_dispatcher) destroyDispatcherPtr_;
    decltype(&avstp_enqueue_task) enqueuePtr_;
    decltype(&avstp_wait_completion) waitPtr_;


	void * dllHandle_;	// Avoids loading windows.h just for HMODULE

	static int dummy_dispatcher_;

    //singleton
	AvstpWrapper(const AvstpWrapper &other);
	AvstpWrapper& operator=(const AvstpWrapper &other);
	bool operator==(const AvstpWrapper &other) const;
	bool operator!=(const AvstpWrapper &other) const;

};

#endif