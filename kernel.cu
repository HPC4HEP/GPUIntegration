
    // future from a packaged_task
    std::packaged_task<int()> task([](){ return 42; });
    std::future<int> futureAnswerToLifeUniverseAndEverything = task.get_future();  
    std::thread(std::move(task)).detach(); 
    std::cout << futureAnswerToLifeUniverseAndEverything.get() << std::endl;
