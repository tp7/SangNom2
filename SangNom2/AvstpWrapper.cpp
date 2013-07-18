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



#if defined (_MSC_VER)
	#pragma warning (1 : 4130 4223 4705 4706)
	#pragma warning (4 : 4355 4786 4800)
#endif

#define	NOGDI
#define	NOMINMAX
#define	WIN32_LEAN_AND_MEAN

#include	"AvstpFinder.h"
#include	"AvstpWrapper.h"

#include	"Windows.h"

#include	<stdexcept>

#include	<cassert>


AvstpWrapper::~AvstpWrapper ()
{
	FreeLibrary (reinterpret_cast < ::HMODULE> (dllHandle_));
	dllHandle_ = nullptr;
}


AvstpWrapper& AvstpWrapper::getInstance()
{
    static AvstpWrapper wrapper;
	return wrapper;
}


avstp_TaskDispatcher* AvstpWrapper::createDispatcher()
{
	return createDispatcherPtr_();
}



void AvstpWrapper::destroyDispatcher(avstp_TaskDispatcher *dispatcher)
{
	destroyDispatcherPtr_(dispatcher);
}



int	AvstpWrapper::numberOfThreads() 
{
	return numberOfThreadsPtr_();
}



int	AvstpWrapper::enqueue(avstp_TaskDispatcher *dispatcher, avstp_TaskPtr task, void *userData) {
	return enqueuePtr_(dispatcher, task, userData);
}



int	AvstpWrapper::waitCompletion(avstp_TaskDispatcher *dispatcher)
{
	return waitPtr_(dispatcher);
}




AvstpWrapper::AvstpWrapper() 
    : dllHandle_(AvstpFinder::find_lib ()), createDispatcherPtr_ (nullptr), destroyDispatcherPtr_ (nullptr),
	                                    numberOfThreadsPtr_ (nullptr), enqueuePtr_ (nullptr), waitPtr_ (nullptr) 
{
	if (dllHandle_ == 0)
	{
		OutputDebugString ("AvstpWrapper: cannot find avstp.dll.\nUsage restricted to single threading.\n");
		assign_fallback ();
	}
	else
	{
		assign_normal ();
	}
}




template <class T>
void	AvstpWrapper::resolveName (T &fnc_ptr, const char *name_0)
{
	assert (&fnc_ptr != 0);
	assert (name_0 != 0);
	assert (dllHandle_ != 0);

	fnc_ptr = reinterpret_cast <T> (
		::GetProcAddress (reinterpret_cast < ::HMODULE> (dllHandle_), name_0)
	);
	if (fnc_ptr == 0)
	{
		::FreeLibrary (reinterpret_cast < ::HMODULE> (dllHandle_));
		dllHandle_ = nullptr;
		throw std::runtime_error ("Function missing in avstp.dll.");
	}
}



void	AvstpWrapper::assign_normal ()
{
	resolveName (createDispatcherPtr_,     "avstp_create_dispatcher");
	resolveName (destroyDispatcherPtr_,    "avstp_destroy_dispatcher");
	resolveName (numberOfThreadsPtr_,       "avstp_get_nbr_threads");
	resolveName (enqueuePtr_,          "avstp_enqueue_task");
	resolveName (waitPtr_,       "avstp_wait_completion");
}



void AvstpWrapper::assign_fallback ()
{
	createDispatcherPtr_ = &fallback_create_dispatcher_ptr;
	destroyDispatcherPtr_ = &fallback_destroy_dispatcher_ptr;
	numberOfThreadsPtr_ = &fallback_get_nbr_threads_ptr;
	enqueuePtr_ = &fallback_enqueue_task_ptr;
	waitPtr_ = &fallback_wait_completion_ptr;
}

avstp_TaskDispatcher* AvstpWrapper::fallback_create_dispatcher_ptr ()
{
	return reinterpret_cast<avstp_TaskDispatcher *>(&dummy_dispatcher_);
}



void	AvstpWrapper::fallback_destroy_dispatcher_ptr (avstp_TaskDispatcher *td_ptr)
{
	assert (td_ptr == reinterpret_cast<avstp_TaskDispatcher *>(&dummy_dispatcher_));
}



int	AvstpWrapper::fallback_get_nbr_threads_ptr ()
{
	return (1);
}



int	AvstpWrapper::fallback_enqueue_task_ptr (avstp_TaskDispatcher *td_ptr, avstp_TaskPtr task_ptr, void *user_data_ptr)
{
	int				ret_val = avstp_Err_OK;

	if (   td_ptr != reinterpret_cast<avstp_TaskDispatcher *>(&dummy_dispatcher_)
	    || task_ptr == 0)
	{
		ret_val = avstp_Err_INVALID_ARG;
	}
	else
	{
		task_ptr (td_ptr, user_data_ptr);
	}

	return (ret_val);
}



int	AvstpWrapper::fallback_wait_completion_ptr (avstp_TaskDispatcher *td_ptr)
{
	int				ret_val = avstp_Err_OK;

	if (td_ptr != reinterpret_cast<avstp_TaskDispatcher *>(&dummy_dispatcher_))
	{
		ret_val = avstp_Err_INVALID_ARG;
	}

	return (ret_val);
}

int	AvstpWrapper::dummy_dispatcher_;

